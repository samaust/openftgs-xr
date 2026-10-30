// SPDX-License-Identifier: Apache-2.0

//! The in-process fixture server of the `cpu` lane's listing and download tests (§2.7,
//! Q-1, Q-6, Q-39, Q-40, Q-55), over tiny_http (§2.3).

mod http;

use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::fixtures_dir;
use http::{
    LAST_MODIFIED, RangeAnswer, content_type, if_range_holds, none_match_hits, percent_decode,
    percent_encode, range_answer,
};

/// The directory whose listing the three committed bodies describe, under the root URL.
const LISTING_BASE: &str = "/listing/models/";
/// The player's own manifest under the listing base (§1.5, D-31).
const LISTING_INDEX: &str = "/listing/models/index.json";
/// Threads answering requests, so a slow reader of one response blocks no other.
const WORKERS: usize = 4;
/// How often an idle worker checks for shutdown.
const POLL: Duration = Duration::from_millis(20);
/// The pause between a truncated body and the invalid chunk line that ends it (Q-40).
const TRUNCATION_PAUSE: Duration = Duration::from_millis(50);

/// Which committed listing body the listing base `url_of("listing/models/")` answers with
/// (Q-6, Q-39). The default is `Index`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ServedListing {
    /// The player's `index.json` (`listing/index.json`) answers at `<base>index.json`; the
    /// base itself answers 404.
    #[default]
    Index,
    /// The base answers nginx's JSON listing (`listing/nginx.json`); `<base>index.json` 404.
    NginxJson,
    /// The base answers Apache's HTML listing (`listing/apache.html`); `<base>index.json` 404.
    Html,
}

/// One request the server answered, recorded before its response was written (Q-39).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestRecord {
    /// The method, as sent (`GET`, `HEAD`, …).
    pub method: String,
    /// The request target as sent: percent-encoded, with any query.
    pub path: String,
    /// The request's headers, names and values as sent, in order.
    pub headers: Vec<(String, String)>,
    /// The response's status code.
    pub status: u16,
    /// The body bytes sent: the range's length for a 206, none for `HEAD` and 304, the kept
    /// bytes for a truncated response (Q-40).
    pub body_bytes: u64,
}

impl RequestRecord {
    /// The value of the request's first header named `name`, compared case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// An in-process HTTP server over the verified `tests/fixtures/` directory (§2.7, Q-1,
/// Q-39), listening on `127.0.0.1`; it stops when dropped.
///
/// Every file under `fixtures_dir()` is served at its relative path, with a strong ETag (the
/// quoted SHA-256 of its bytes), a fixed `Last-Modified`, `If-None-Match` and
/// `If-Modified-Since` answered by 304 and a single byte `Range` by 206 (Q-39). The
/// listing base `url_of("listing/models/")` answers with the body `serve_listing` chose.
pub struct LocalServer {
    port: u16,
    state: Arc<State>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

/// What the workers share with the handle.
struct State {
    root: PathBuf,
    listing: Mutex<ServedListing>,
    truncate: Mutex<Option<u64>>,
    statuses: Mutex<HashMap<String, u16>>,
    records: Mutex<Vec<RequestRecord>>,
}

/// A new fixture server on a free port of `127.0.0.1` (§2.7, Q-1, Q-39).
///
/// # Panics
///
/// When the fixtures fail their checksum check (`fixtures_dir`) or no port can be bound.
pub fn serve_fixtures() -> LocalServer {
    let root = fixtures_dir();
    let server = Arc::new(
        Server::http("127.0.0.1:0").unwrap_or_else(|e| panic!("fixture server: bind: {e}")),
    );
    let port = server
        .server_addr()
        .to_ip()
        .expect("fixture server: a TCP address")
        .port();
    let state = Arc::new(State {
        root,
        listing: Mutex::new(ServedListing::default()),
        truncate: Mutex::new(None),
        statuses: Mutex::new(HashMap::new()),
        records: Mutex::new(Vec::new()),
    });
    let stop = Arc::new(AtomicBool::new(false));
    let workers = (0..WORKERS)
        .map(|_| {
            let (server, state, stop) = (server.clone(), state.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Ok(Some(request)) = server.recv_timeout(POLL) {
                        answer(&state, request);
                    }
                }
            })
        })
        .collect();
    LocalServer {
        port,
        state,
        stop,
        workers,
    }
}

impl LocalServer {
    /// The server's root URL, `http://127.0.0.1:<port>/`.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// The URL of the fixture at `path`, relative to `fixtures_dir()` (`/`-separated), with
    /// every byte outside RFC 3986's unreserved set percent-encoded. The listing base is
    /// `url_of("listing/models/")`.
    pub fn url_of(&self, path: &str) -> String {
        format!(
            "{}{}",
            self.url(),
            percent_encode(path.trim_start_matches('/'))
        )
    }

    /// Every request answered so far, in arrival order.
    pub fn requests(&self) -> Vec<RequestRecord> {
        lock(&self.state.records).clone()
    }

    /// Truncates the next response that carries a body (a 200 or 206 to a `GET`) after
    /// `keep` bytes of it: the response is sent chunked, and the kept bytes are followed by
    /// an invalid chunk line, so the client's read of the body fails (Q-40). One call
    /// arms one truncation; a later call replaces an unused one.
    pub fn truncate_next(&self, keep: u64) {
        *lock(&self.state.truncate) = Some(keep);
    }

    /// Chooses which committed listing body the listing base answers with from now on
    /// (Q-6, Q-39).
    pub fn serve_listing(&self, kind: ServedListing) {
        *lock(&self.state.listing) = kind;
    }

    /// Makes every later GET or HEAD of the request target `path` (as sent, percent-encoded,
    /// its query ignored: `/listing/models/`) answer `status` with an empty body, before
    /// the listing routes and the files; a later call for `path` replaces the status
    /// (Q-55).
    // Q-55
    pub fn status_for(&self, path: &str, status: u16) {
        lock(&self.state.statuses).insert(path.to_owned(), status);
    }
}

impl Drop for LocalServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// The guarded value, whether or not a worker panicked while holding it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A response decided but not yet written.
struct Answer {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Answer {
    fn text(status: u16, text: &str) -> Answer {
        Answer {
            status,
            headers: vec![("Content-Type", content_type(".txt").to_owned())],
            body: text.as_bytes().to_vec(),
        }
    }
}

/// Answers one request: records it, then writes the response, truncated when armed.
fn answer(state: &State, request: Request) {
    let method = request.method().clone();
    let answer = respond_to(state, &request);
    let head = method == Method::Head;
    let truncation = if !head && method == Method::Get && matches!(answer.status, 200 | 206) {
        lock(&state.truncate).take()
    } else {
        None
    };
    let body_bytes = match (head, truncation) {
        (true, _) => 0,
        (false, Some(keep)) => keep.min(answer.body.len() as u64),
        (false, None) if answer.status == 304 => 0,
        (false, None) => answer.body.len() as u64,
    };
    lock(&state.records).push(RequestRecord {
        method: method.as_str().to_owned(),
        path: request.url().to_owned(),
        headers: request
            .headers()
            .iter()
            .map(|h| (h.field.as_str().to_string(), h.value.as_str().to_owned()))
            .collect(),
        status: answer.status,
        body_bytes,
    });
    // A failed write means the client went away; the test that drove it sees that itself.
    let _ = match truncation {
        Some(_) => write_truncated(request, &answer, body_bytes as usize),
        None => write_whole(request, answer),
    };
}

/// Writes the response through tiny_http, always with a `Content-Length`.
fn write_whole(request: Request, answer: Answer) -> std::io::Result<()> {
    let length = answer.body.len();
    let headers = answer
        .headers
        .iter()
        .map(|(name, value)| {
            Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("an ASCII header")
        })
        .collect();
    let response = Response::new(
        StatusCode(answer.status),
        headers,
        Cursor::new(answer.body),
        Some(length),
        None,
    )
    .with_chunked_threshold(usize::MAX);
    request.respond(response)
}

/// Writes the response raw: chunked, the first `keep` body bytes as one chunk, a pause,
/// then an invalid chunk-size line (Q-40).
fn write_truncated(request: Request, answer: &Answer, keep: usize) -> std::io::Result<()> {
    let reason = StatusCode(answer.status).default_reason_phrase();
    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 {} {reason}\r\n", answer.status)?;
    for (name, value) in &answer.headers {
        write!(writer, "{name}: {value}\r\n")?;
    }
    write!(writer, "Transfer-Encoding: chunked\r\n\r\n")?;
    if keep > 0 {
        write!(writer, "{keep:x}\r\n")?;
        writer.write_all(&answer.body[..keep])?;
        write!(writer, "\r\n")?;
    }
    writer.flush()?;
    std::thread::sleep(TRUNCATION_PAUSE);
    write!(writer, "truncated by the fixture server\r\n")?;
    writer.flush()
}

/// Decides the response to `request` (Q-39).
fn respond_to(state: &State, request: &Request) -> Answer {
    let method = request.method();
    if *method != Method::Get && *method != Method::Head {
        let mut answer = Answer::text(405, "method not allowed\n");
        answer.headers.push(("Allow", "GET, HEAD".to_owned()));
        return answer;
    }
    let target = request.url();
    let path = target.split_once('?').map_or(target, |(path, _)| path);
    // Q-55
    if let Some(&status) = lock(&state.statuses).get(path) {
        return Answer {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        };
    }
    let Some(file) = percent_decode(path).and_then(|path| resolve(state, &path)) else {
        return Answer::text(404, "not found\n");
    };
    let Ok(body) = std::fs::read(&file) else {
        return Answer::text(404, "not found\n");
    };
    let etag = format!("\"{}\"", ftgsxr_core::hex(&ftgsxr_core::sha256(&body)));
    let header = |name: &'static str| request_header(request, name);
    let mut headers = vec![
        ("ETag", etag.clone()),
        ("Last-Modified", LAST_MODIFIED.to_owned()),
    ];
    let not_modified = match header("If-None-Match") {
        Some(tags) => none_match_hits(tags, &etag),
        None => header("If-Modified-Since").is_some_and(|date| date.trim() == LAST_MODIFIED),
    };
    if not_modified {
        return Answer {
            status: 304,
            headers,
            body: Vec::new(),
        };
    }
    let name = file
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    headers.push(("Content-Type", content_type(&name).to_owned()));
    headers.push(("Accept-Ranges", "bytes".to_owned()));
    let len = body.len() as u64;
    let range = match header("Range") {
        Some(range)
            if *method == Method::Get
                && header("If-Range").is_none_or(|value| if_range_holds(value, &etag)) =>
        {
            range_answer(range, len)
        }
        _ => RangeAnswer::Whole,
    };
    match range {
        RangeAnswer::Whole => Answer {
            status: 200,
            headers,
            body,
        },
        RangeAnswer::Part { start, end } => {
            headers.push(("Content-Range", format!("bytes {start}-{end}/{len}")));
            Answer {
                status: 206,
                headers,
                body: body[start as usize..=end as usize].to_vec(),
            }
        }
        RangeAnswer::Unsatisfiable => {
            let mut answer = Answer::text(416, "range not satisfiable\n");
            answer
                .headers
                .push(("Content-Range", format!("bytes */{len}")));
            answer
        }
    }
}

/// The value of the request's first header named `name` (case-insensitive).
fn request_header<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str())
}

/// The file a decoded request path names (Q-39): the listing routes, else the regular
/// file at that path under the fixtures root; `None` for a directory, a missing file or a
/// path with an empty, `.` or `..` segment.
fn resolve(state: &State, path: &str) -> Option<PathBuf> {
    let listing = *lock(&state.listing);
    let routed = match path {
        LISTING_BASE => match listing {
            ServedListing::Index => return None,
            ServedListing::NginxJson => Some("listing/nginx.json"),
            ServedListing::Html => Some("listing/apache.html"),
        },
        LISTING_INDEX => match listing {
            ServedListing::Index => Some("listing/index.json"),
            _ => return None,
        },
        _ => None,
    };
    if let Some(relative) = routed {
        return Some(state.root.join(relative));
    }
    let relative = path.strip_prefix('/')?;
    if relative
        .split('/')
        .any(|segment| matches!(segment, "" | "." | ".."))
        || relative.contains('\\')
    {
        return None;
    }
    let file = state.root.join(Path::new(relative));
    file.is_file().then_some(file)
}
