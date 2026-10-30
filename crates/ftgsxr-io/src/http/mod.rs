// SPDX-License-Identifier: Apache-2.0

//! The HTTP client: listings as text, header-only range reads, and the download cache's
//! cached, resumable, validated downloads (§1.5, §1.11, D-20).

mod cache;
mod download;
mod prune;

use std::collections::HashSet;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use ureq::Agent;
use ureq::http::Response;
use ureq::tls::{RootCerts, TlsConfig, TlsProvider};
use url::Url;

use crate::IoError;

/// How long a connection (with its TLS handshake) may take before E-NET-FETCH (Q-43).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// How long the response headers may take once the request is sent (Q-43).
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long one read of the body may stall (Q-43).
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// The largest listing body `get_text` reads (Q-44).
const TEXT_LIMIT: u64 = 64 << 20;

/// The root certificates TLS connections are verified against (D-20).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsRoots {
    /// Mozilla's roots, bundled through webpki-roots (D-20); both builds' only roots.
    WebPki,
}

/// A download's progress (§1.5): the bytes of the file held so far, and its size when
/// the server gave one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Bytes of the file received or resumed so far.
    pub done: u64,
    /// The file's whole size, when known.
    pub total: Option<u64>,
}

/// The player's HTTP client (§1.5, D-20): one ureq agent over rustls with the `ring`
/// provider, and the download cache under `<cache dir>/http/` (§1.11, Q-42).
///
/// The client is `Send + Sync`, shared through `Arc`; downloads and `prune` take one lock,
/// so they run one at a time (Q-45), while text and range reads run in parallel.
pub struct HttpClient {
    agent: Agent,
    /// `<cache dir>/http`, the download cache's entries (Q-42).
    entries: PathBuf,
    max_bytes: u64,
    /// The cached files `prune` keeps (Q-4).
    in_use: Mutex<HashSet<PathBuf>>,
    /// Held by `download` and `prune` (Q-45).
    cache_lock: Mutex<()>,
}

impl HttpClient {
    /// A client whose download cache lives under `cache_dir/http/` (§1.11 Locations and
    /// Cache entries, Q-42), pruned to `max_bytes` by `prune` (§1.11 `[cache]`).
    ///
    /// E-IO when the cache directory cannot be created.
    pub fn new(cache_dir: PathBuf, roots: TlsRoots, max_bytes: u64) -> Result<HttpClient, IoError> {
        let TlsRoots::WebPki = roots;
        // D-20: rustls (ureq's `rustls` feature: the `ring` provider) with the webpki roots.
        let tls = TlsConfig::builder()
            .provider(TlsProvider::Rustls)
            .root_certs(RootCerts::WebPki)
            .build();
        let agent = Agent::config_builder()
            .tls_config(tls)
            .http_status_as_error(false)
            .user_agent(concat!("openftgs-xr/", env!("CARGO_PKG_VERSION")))
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_recv_response(Some(RESPONSE_TIMEOUT))
            .timeout_per_read(Some(READ_TIMEOUT))
            .build()
            .new_agent();
        let entries = cache_dir.join("http");
        std::fs::create_dir_all(&entries)?;
        Ok(HttpClient {
            agent,
            entries,
            max_bytes,
            in_use: Mutex::new(HashSet::new()),
            cache_lock: Mutex::new(()),
        })
    }
}

impl HttpClient {
    /// The body of a 2xx answer to a GET of `url`, as text (§1.5's listings): uncached,
    /// with invalid UTF-8 replaced (Q-44). E-NET-FETCH on any other answer.
    pub fn get_text(&self, url: &Url) -> Result<String, IoError> {
        self.get_text_or_status(url)?
            .map_err(|status| status_error(url, status))
    }

    /// `get_text`, with a status other than 2xx returned as `Ok(Err(status))` rather than
    /// E-NET-FETCH, so a caller can tell "the server answered no" from "the server could
    /// not be reached" (`ListingFlavor::Auto`'s detection, Q-48).
    pub(crate) fn get_text_or_status(&self, url: &Url) -> Result<Result<String, u16>, IoError> {
        let mut response = self
            .agent
            .get(url.as_str())
            .call()
            .map_err(|e| fetch_error(url, &e))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Ok(Err(status));
        }
        response
            .body_mut()
            .with_config()
            .limit(TEXT_LIMIT)
            .lossy_utf8(true)
            .read_to_string()
            .map(Ok)
            .map_err(|e| fetch_error(url, &e))
    }

    /// At most `len` bytes of the file at `url` from byte `start` on, through one `Range`
    /// request (§1.5's header-only reads: the first 8 + header-length bytes, never the
    /// whole file). Fewer bytes when the file ends first, none when `start` is at or past
    /// its end (a 416); a server that ignores `Range` is read only up to `start + len` bytes
    /// (Q-44). E-NET-FETCH on any other answer.
    pub fn get_range(&self, url: &Url, start: u64, len: u64) -> Result<Vec<u8>, IoError> {
        self.get_range_sized(url, start, len)
            .map(|(bytes, _)| bytes)
    }

    /// `get_range`, with the whole file's size when the answer states it: a 206's or a 416's
    /// `Content-Range` total, a 200's `Content-Length` (the header-only information's
    /// file size for a listing that gives none, Q-47). `len` 0 sends no request and
    /// knows no size.
    pub(crate) fn get_range_sized(
        &self,
        url: &Url,
        start: u64,
        len: u64,
    ) -> Result<(Vec<u8>, Option<u64>), IoError> {
        if len == 0 {
            return Ok((Vec::new(), None));
        }
        let last = start.saturating_add(len - 1);
        let response = self
            .agent
            .get(url.as_str())
            .header("Range", format!("bytes={start}-{last}"))
            .call()
            .map_err(|e| fetch_error(url, &e))?;
        let (skip, size) = match response.status().as_u16() {
            206 => match content_range(&response) {
                Some((first, size)) if first == start => (0, size),
                _ => return Err(IoError::Fetch(format!("{url} (unexpected Content-Range)"))),
            },
            // Q-44: a server without ranges; read only the prefix that holds the slice.
            200 => (
                start,
                header(&response, "Content-Length").and_then(|v| v.trim().parse().ok()),
            ),
            // Q-44 (changed, #87): `start` is at or past the end of the file, so no byte.
            416 => return Ok((Vec::new(), unsatisfied_size(&response))),
            status => return Err(status_error(url, status)),
        };
        let mut body = Vec::new();
        response
            .into_body()
            .into_reader()
            .take(skip.saturating_add(len))
            .read_to_end(&mut body)
            .map_err(|e| IoError::Fetch(format!("{url} ({e})")))?;
        Ok((body.split_off((skip as usize).min(body.len())), size))
    }
}

/// E-NET-FETCH for a request that failed before a status arrived (§1.15).
fn fetch_error(url: &Url, error: &ureq::Error) -> IoError {
    IoError::Fetch(format!("{url} ({error})"))
}

/// E-NET-FETCH for an unexpected status (§1.15: "Could not reach URL (status)").
fn status_error(url: &Url, status: u16) -> IoError {
    IoError::Fetch(format!("{url} (status {status})"))
}

/// The value of the response's header `name`, when it is text.
fn header<'a, B>(response: &'a Response<B>, name: &str) -> Option<&'a str> {
    response.headers().get(name).and_then(|v| v.to_str().ok())
}

/// A 206's `Content-Range: bytes <first>-<last>/<size or *>` as (first, size).
fn content_range<B>(response: &Response<B>) -> Option<(u64, Option<u64>)> {
    let value = header(response, "Content-Range")?.trim();
    let (range, size) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (first, _) = range.split_once('-')?;
    let size = match size.trim() {
        "*" => None,
        size => Some(size.parse().ok()?),
    };
    Some((first.trim().parse().ok()?, size))
}

/// A 416's `Content-Range: bytes */<size>` as the file's size.
fn unsatisfied_size<B>(response: &Response<B>) -> Option<u64> {
    let value = header(response, "Content-Range")?.trim();
    value.strip_prefix("bytes */")?.trim().parse().ok()
}

/// The guarded value, whether or not a thread panicked while holding it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    /// §2.1 shares the client through `Arc` (`open_source`), across threads.
    #[test]
    fn http_client_is_send_and_sync() {
        fn shared<T: Send + Sync>() {}
        shared::<super::HttpClient>();
    }

    /// Q-44 (#87): a 416 gives no byte and the size its `Content-Range: bytes */<size>`
    /// states (the fixture server's, Q-39).
    #[test]
    fn get_range_sized_416_gives_the_stated_size() {
        let server = ftgsxr_testkit::serve_fixtures();
        let dir = std::env::temp_dir().join("ftgsxr-http-unit-416");
        let _ = std::fs::remove_dir_all(&dir);
        let http = super::HttpClient::new(dir, super::TlsRoots::WebPki, 1 << 30).unwrap();
        let name = "checkpoint/tiny.safetensors";
        let size = std::fs::metadata(ftgsxr_testkit::fixtures_dir().join(name))
            .unwrap()
            .len();
        let url = url::Url::parse(&server.url_of(name)).unwrap();
        let answer = http.get_range_sized(&url, size, 8).map_err(|e| e.code());
        assert_eq!(answer, Ok((Vec::new(), Some(size))));
    }
}
