// SPDX-License-Identifier: Apache-2.0

//! `HttpClient::download`: the download cache's cached, resumable, validated downloads
//! (§1.5 Sources and Download cache, §1.11 Cache entries, Q-T18.1, Q-T18.2).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};
use ureq::Body;
use ureq::http::Response;
use url::Url;

use super::cache::{Entry, Record, now_rfc3339, remove_if_present};
use super::{HttpClient, Progress, content_range, fetch_error, header, lock, status_error};
use crate::IoError;

/// The bytes read from the body between two cancel checks and progress reports: "one
/// chunk" of U-28's cancel clause.
const CHUNK: usize = 64 << 10;

/// What a request asks of the server.
enum Ask {
    /// The whole file, unconditionally.
    Fresh,
    /// The whole file unless the complete cached copy still matches its validators (§1.5).
    Revalidate(Record),
    /// The rest of a partial download from byte `held`, if its validators still match (§1.5).
    Resume(Record, u64),
}

impl HttpClient {
    /// A cached, resumable, validated download (§1.5); returns the cached file's path.
    ///
    /// The file lands in its cache entry `<cache dir>/http/<sha256 of the URL>/file`
    /// (§1.11, Q-T18.1), written to a temporary sibling and renamed on completion. A
    /// complete entry is revalidated with `If-None-Match` and `If-Modified-Since` and reused
    /// on a 304; a partial one is resumed with `Range` and `If-Range`, or restarted when its
    /// validators no longer match (§1.5, Q-T18.2). With `expected_sha256` the finished file
    /// must have that digest, or it is discarded with E-NET-CHECKSUM (§1.5, §1.15).
    /// `progress` hears after every chunk; `cancel` is read before every chunk, and a
    /// cancelled download keeps its partial file and returns E-NET-FETCH (Q-T18.2).
    pub fn download(
        &self,
        url: &Url,
        expected_sha256: Option<&str>,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<PathBuf, IoError> {
        let expected = match expected_sha256 {
            None => None,
            // Q-T18.2: a malformed digest is refused before any request.
            Some(digest) => Some(parse_digest(digest).ok_or_else(|| {
                IoError::Checksum(format!("{url}: expected digest {digest:?} is not SHA-256"))
            })?),
        };
        let _guard = lock(&self.cache_lock);
        let entry = Entry::of(&self.entries, url);
        std::fs::create_dir_all(&entry.dir)?;
        let complete = entry
            .complete(Some(url))
            // Q-T18.2: a digest from the index wins over the cache's record.
            .filter(|record| expected.is_none() || record.sha256 == expected);
        let mut ask = match (entry.partial(Some(url)), complete) {
            (Some((record, held)), _)
                if held > 0 && (record.etag.is_some() || record.last_modified.is_some()) =>
            {
                Ask::Resume(record, held)
            }
            (_, Some(record)) => Ask::Revalidate(record),
            (_, None) => Ask::Fresh,
        };
        // One fresh request after a refused resume (Q-T18.2).
        for _ in 0..2 {
            if cancel.load(Ordering::Relaxed) {
                return Err(cancelled(url));
            }
            let response = self.send(url, &ask)?;
            let status = response.status().as_u16();
            match (status, ask) {
                (304, Ask::Revalidate(mut record)) => {
                    // Q-T18.1: a 304 refreshes the fetch time, the entry's last use.
                    record.fetched_at = now_rfc3339();
                    record.write(&entry.meta)?;
                    let size = record.size;
                    progress(Progress {
                        done: size.unwrap_or(0),
                        total: size,
                    });
                    return Ok(entry.file);
                }
                (206, Ask::Resume(record, held)) => match content_range(&response) {
                    Some((first, total)) if first == held => {
                        let total = total.or(record.size);
                        return self.receive(
                            url, &entry, record, held, total, response, expected, progress, cancel,
                        );
                    }
                    _ => {
                        entry.discard_partial()?;
                        ask = Ask::Fresh;
                    }
                },
                (416, Ask::Resume(..)) => {
                    entry.discard_partial()?;
                    ask = Ask::Fresh;
                }
                (200, _) => {
                    let record = Record {
                        url: url.as_str().to_owned(),
                        etag: header(&response, "ETag").map(str::to_owned),
                        last_modified: header(&response, "Last-Modified").map(str::to_owned),
                        size: response.body().content_length(),
                        sha256: None,
                        fetched_at: now_rfc3339(),
                    };
                    entry.discard_partial()?;
                    record.write(&entry.part_meta)?;
                    let total = record.size;
                    return self.receive(
                        url, &entry, record, 0, total, response, expected, progress, cancel,
                    );
                }
                (status, _) => return Err(status_error(url, status)),
            }
        }
        Err(IoError::Fetch(format!(
            "{url} (the server refused to resume)"
        )))
    }

    /// Sends the request `ask` describes.
    fn send(&self, url: &Url, ask: &Ask) -> Result<Response<Body>, IoError> {
        let mut request = self.agent.get(url.as_str());
        match ask {
            Ask::Fresh => {}
            Ask::Revalidate(record) => {
                if let Some(etag) = &record.etag {
                    request = request.header("If-None-Match", etag);
                }
                if let Some(date) = &record.last_modified {
                    request = request.header("If-Modified-Since", date);
                }
            }
            Ask::Resume(record, held) => {
                let validator = record.etag.as_ref().or(record.last_modified.as_ref());
                request = request.header("Range", format!("bytes={held}-"));
                if let Some(validator) = validator {
                    request = request.header("If-Range", validator);
                }
            }
        }
        request.call().map_err(|e| fetch_error(url, &e))
    }

    /// Streams the body into `file.part` after its first `held` bytes, then checks it and
    /// completes the entry (§1.5, §1.11, Q-T18.2).
    #[allow(clippy::too_many_arguments)]
    fn receive(
        &self,
        url: &Url,
        entry: &Entry,
        mut record: Record,
        held: u64,
        total: Option<u64>,
        response: Response<Body>,
        expected: Option<String>,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<PathBuf, IoError> {
        let mut hasher = Sha256::new();
        let mut file = if held == 0 {
            File::create(&entry.part)?
        } else {
            // The digest covers the bytes already held (a hasher cannot outlive a process).
            let mut part = File::open(&entry.part)?;
            let copied = std::io::copy(&mut (&mut part).take(held), &mut hasher)?;
            if copied != held {
                entry.discard_partial()?;
                return Err(IoError::Fetch(format!(
                    "{url} (the partial download shrank)"
                )));
            }
            let mut file = OpenOptions::new().write(true).open(&entry.part)?;
            file.set_len(held)?;
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(held))?;
            file
        };
        let mut reader = response.into_body().into_reader();
        let mut buf = vec![0u8; CHUNK];
        let mut done = held;
        loop {
            if cancel.load(Ordering::Relaxed) {
                file.flush()?;
                return Err(cancelled(url));
            }
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    // Q-T18.2: keep every byte received; the next call resumes.
                    file.flush()?;
                    return Err(IoError::Fetch(format!("{url} ({e})")));
                }
            };
            file.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            done += n as u64;
            progress(Progress { done, total });
        }
        file.sync_all()?;
        drop(file);
        if total.is_some_and(|total| total != done) {
            return Err(IoError::Fetch(format!("{url} (the body ended early)")));
        }
        let digest = ftgsxr_core::hex(&hasher.finalize().into());
        if expected
            .as_ref()
            .is_some_and(|expected| *expected != digest)
        {
            entry.discard_partial()?;
            return Err(IoError::Checksum(url.to_string()));
        }
        std::fs::rename(&entry.part, &entry.file)?;
        record.size = Some(done);
        record.sha256 = Some(digest);
        record.fetched_at = now_rfc3339();
        record.write(&entry.meta)?;
        remove_if_present(&entry.part_meta)?;
        Ok(entry.file.clone())
    }
}

/// The cancel error (Q-T18.2).
fn cancelled(url: &Url) -> IoError {
    IoError::Fetch(format!("{url}: cancelled"))
}

/// A SHA-256 given as 64 hex digits in either case, lowercased (Q-T18.2).
fn parse_digest(digest: &str) -> Option<String> {
    let digest = digest.trim();
    (digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| digest.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::TlsRoots;

    /// §1.5: "resumed with a `Range` request when the validators still match, else
    /// restarted". The partial's validator is made stale, so the fixture server ignores the
    /// range (`If-Range`, Q-39) and the client restarts from the first byte.
    #[test]
    fn download_restarts_when_the_validators_changed() {
        let server = ftgsxr_testkit::serve_fixtures();
        let cache = std::env::temp_dir().join(format!("ftgsxr-restart-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, u64::MAX).unwrap();
        let name = "probes/P8.renders.safetensors";
        let url = Url::parse(&server.url_of(name)).unwrap();
        let whole = std::fs::read(ftgsxr_testkit::fixtures_dir().join(name)).unwrap();
        let cancel = AtomicBool::new(false);

        server.truncate_next(100_000);
        assert!(http.download(&url, None, &mut |_| {}, &cancel).is_err());
        let entry = Entry::of(&http.entries, &url);
        let (mut record, held) = entry.partial(Some(&url)).expect("a partial download");
        assert_eq!(held, 100_000);
        record.etag = Some("\"stale\"".into());
        record.last_modified = None;
        record.write(&entry.part_meta).unwrap();

        let path = http.download(&url, None, &mut |_| {}, &cancel).unwrap();
        assert_eq!(std::fs::read(path).unwrap(), whole);
        let last = server.requests().pop().unwrap();
        assert_eq!(last.header("If-Range"), Some("\"stale\""));
        assert_eq!((last.status, last.body_bytes), (200, whole.len() as u64));
        assert!(!entry.part.exists() && !entry.part_meta.exists());
        let _ = std::fs::remove_dir_all(&cache);
    }
}
