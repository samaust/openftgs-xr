// SPDX-License-Identifier: Apache-2.0

//! §2.7 `serve_fixtures`, Q-1, Q-6, Q-39, Q-40: the fixture server answers whole-file,
//! 304 and 206 requests, records them, truncates on demand and serves the chosen listing
//! body, as U-28's clauses and §1.5's cache paragraph need. Driven over a raw socket, so
//! every header and byte on the wire is visible.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use ftgsxr_testkit::{LocalServer, ServedListing, fixtures_dir, serve_fixtures};

/// The listing's model with a space in its name.
const MODEL: &str = "listing/models/probe p1.safetensors";
/// Its path as a client sends it.
const MODEL_TARGET: &str = "/listing/models/probe%20p1.safetensors";
const LAST_MODIFIED: &str = "Thu, 29 Oct 2026 12:00:00 GMT";

/// A response as read off the socket.
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    /// The body after any chunked framing was removed, up to where the framing broke.
    body: Vec<u8>,
    /// Whether the chunked framing ended with an invalid chunk line.
    broken: bool,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Sends one HTTP/1.1 request with `Connection: close` and reads the reply to EOF.
fn send(server: &LocalServer, method: &str, target: &str, headers: &[(&str, &str)]) -> Reply {
    let address = server.url();
    let address = address
        .strip_prefix("http://")
        .unwrap()
        .trim_end_matches('/');
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut request =
        format!("{method} {target} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = std::str::from_utf8(&raw[..split]).unwrap();
    let rest = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers: Vec<(String, String)> = lines
        .map(|line| {
            let (n, v) = line.split_once(':').unwrap();
            (n.trim().to_owned(), v.trim().to_owned())
        })
        .collect();
    let mut reply = Reply {
        status,
        headers,
        body: Vec::new(),
        broken: false,
    };
    if reply.header("Transfer-Encoding") == Some("chunked") {
        let (body, broken) = dechunk(rest);
        reply.body = body;
        reply.broken = broken;
    } else {
        let len: usize = reply.header("Content-Length").unwrap().parse().unwrap();
        if method != "HEAD" {
            assert_eq!(rest.len(), len, "the body is the Content-Length");
            reply.body = rest.to_vec();
        } else {
            assert!(rest.is_empty());
        }
    }
    reply
}

/// The data of a chunked body, and whether a chunk-size line failed to parse.
fn dechunk(mut rest: &[u8]) -> (Vec<u8>, bool) {
    let mut body = Vec::new();
    loop {
        let Some(end) = rest.windows(2).position(|w| w == b"\r\n") else {
            return (body, true);
        };
        let Ok(size) = usize::from_str_radix(std::str::from_utf8(&rest[..end]).unwrap(), 16) else {
            return (body, true);
        };
        if size == 0 {
            return (body, false);
        }
        body.extend_from_slice(&rest[end + 2..end + 2 + size]);
        rest = &rest[end + 4 + size..];
    }
}

fn fixture(path: &str) -> Vec<u8> {
    std::fs::read(fixtures_dir().join(path)).unwrap()
}

/// The quoted lowercase hex SHA-256 of `bytes` (Q-39's ETag).
fn etag_of(bytes: &[u8]) -> String {
    format!("\"{}\"", ftgsxr_core::hex(&ftgsxr_core::sha256(bytes)))
}

#[test]
fn url_is_the_root_and_url_of_percent_encodes() {
    let server = serve_fixtures();
    let url = server.url();
    assert!(
        url.starts_with("http://127.0.0.1:") && url.ends_with('/'),
        "{url}"
    );
    assert_eq!(
        server.url_of(MODEL),
        format!("{}{}", url, &MODEL_TARGET[1..])
    );
    assert_eq!(
        server.url_of("listing/models/"),
        format!("{url}listing/models/")
    );
}

#[test]
fn a_whole_file_answers_200_with_its_validators() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    let reply = send(&server, "GET", MODEL_TARGET, &[]);
    assert_eq!(reply.status, 200);
    assert_eq!(reply.body, bytes);
    assert_eq!(reply.header("ETag"), Some(etag_of(&bytes).as_str()));
    assert_eq!(reply.header("Last-Modified"), Some(LAST_MODIFIED));
    assert_eq!(reply.header("Accept-Ranges"), Some("bytes"));
    assert_eq!(
        reply.header("Content-Type"),
        Some("application/octet-stream")
    );
}

#[test]
fn every_fixture_is_served_whole_at_its_path() {
    let server = serve_fixtures();
    for path in [
        "probes/P1.request.json",
        "checkpoint/tiny/params.safetensors",
        "README.md",
    ] {
        let target = server.url_of(path)[server.url().len() - 1..].to_owned();
        let reply = send(&server, "GET", &target, &[]);
        assert_eq!((reply.status, reply.body), (200, fixture(path)), "{path}");
    }
}

#[test]
fn a_matching_if_none_match_answers_304_without_a_body() {
    let server = serve_fixtures();
    let etag = etag_of(&fixture(MODEL));
    let tags = [
        etag.clone(),
        format!("W/{etag}"),
        format!("\"other\", {etag}"),
        "*".to_owned(),
    ];
    for tag in tags {
        let reply = send(&server, "GET", MODEL_TARGET, &[("If-None-Match", &tag)]);
        assert_eq!(reply.status, 304, "{tag}");
        assert!(reply.body.is_empty());
        assert_eq!(reply.header("ETag"), Some(etag.as_str()));
    }
    let reply = send(
        &server,
        "GET",
        MODEL_TARGET,
        &[("If-None-Match", "\"other\"")],
    );
    assert_eq!(reply.status, 200);
}

#[test]
fn if_modified_since_answers_304_unless_if_none_match_differs() {
    let server = serve_fixtures();
    let reply = send(
        &server,
        "GET",
        MODEL_TARGET,
        &[("If-Modified-Since", LAST_MODIFIED)],
    );
    assert_eq!(reply.status, 304);
    let reply = send(
        &server,
        "GET",
        MODEL_TARGET,
        &[
            ("If-None-Match", "\"other\""),
            ("If-Modified-Since", LAST_MODIFIED),
        ],
    );
    assert_eq!(reply.status, 200, "If-None-Match takes precedence");
}

#[test]
fn a_byte_range_answers_206_with_its_content_range() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    let len = bytes.len();
    let cases = [
        ("bytes=0-7", 0, 7),
        ("bytes=100-", 100, len - 1),
        ("bytes=-10", len - 10, len - 1),
        ("bytes=8-99999", 8, len - 1),
    ];
    for (range, start, end) in cases {
        let reply = send(&server, "GET", MODEL_TARGET, &[("Range", range)]);
        assert_eq!(reply.status, 206, "{range}");
        assert_eq!(reply.body, bytes[start..=end], "{range}");
        assert_eq!(
            reply.header("Content-Range"),
            Some(format!("bytes {start}-{end}/{len}").as_str())
        );
    }
}

#[test]
fn a_range_past_the_end_answers_416_and_other_ranges_the_whole_file() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    let len = bytes.len();
    let reply = send(
        &server,
        "GET",
        MODEL_TARGET,
        &[("Range", &format!("bytes={len}-"))],
    );
    assert_eq!(reply.status, 416);
    assert_eq!(
        reply.header("Content-Range"),
        Some(format!("bytes */{len}").as_str())
    );
    for range in ["bytes=0-1,4-5", "bytes=5-2", "lines=1-2"] {
        let reply = send(&server, "GET", MODEL_TARGET, &[("Range", range)]);
        assert_eq!((reply.status, reply.body.len()), (200, len), "{range}");
    }
}

#[test]
fn if_range_resumes_only_while_the_validators_hold() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    let etag = etag_of(&bytes);
    for validator in [etag.as_str(), LAST_MODIFIED] {
        let reply = send(
            &server,
            "GET",
            MODEL_TARGET,
            &[("Range", "bytes=10-"), ("If-Range", validator)],
        );
        assert_eq!((reply.status, reply.body), (206, bytes[10..].to_vec()));
    }
    for stale in ["\"stale\"", "Fri, 30 Oct 2026 12:00:00 GMT"] {
        let reply = send(
            &server,
            "GET",
            MODEL_TARGET,
            &[("Range", "bytes=10-"), ("If-Range", stale)],
        );
        assert_eq!((reply.status, &reply.body), (200, &bytes), "{stale}");
    }
}

#[test]
fn head_answers_the_headers_alone() {
    let server = serve_fixtures();
    let len = fixture(MODEL).len();
    let reply = send(&server, "HEAD", MODEL_TARGET, &[]);
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.header("Content-Length"),
        Some(len.to_string().as_str())
    );
    assert!(reply.body.is_empty());
    assert_eq!(server.requests()[0].body_bytes, 0);
}

#[test]
fn requests_are_recorded_in_order_with_their_headers_status_and_bytes() {
    let server = serve_fixtures();
    send(&server, "GET", MODEL_TARGET, &[]);
    // A header-only read (U-28, U-31): the first 8 bytes, then the header.
    send(&server, "GET", MODEL_TARGET, &[("Range", "bytes=0-7")]);
    send(
        &server,
        "GET",
        "/listing/models/missing.safetensors?x=1",
        &[],
    );
    let records = server.requests();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].method, "GET");
    assert_eq!(records[0].path, MODEL_TARGET);
    assert_eq!(records[0].status, 200);
    assert_eq!(records[0].body_bytes, fixture(MODEL).len() as u64);
    assert_eq!(records[0].header("range"), None);
    assert_eq!(records[1].header("range"), Some("bytes=0-7"));
    assert_eq!((records[1].status, records[1].body_bytes), (206, 8));
    assert_eq!(records[2].path, "/listing/models/missing.safetensors?x=1");
    assert_eq!(records[2].status, 404);
    assert_eq!(records[2].header("CONNECTION"), Some("close"));
}

#[test]
fn truncate_next_breaks_the_next_body_after_the_kept_bytes() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    server.truncate_next(300);
    // Neither a 404 nor a 304 uses the truncation.
    assert_eq!(send(&server, "GET", "/missing", &[]).status, 404);
    let etag = etag_of(&bytes);
    assert_eq!(
        send(&server, "GET", MODEL_TARGET, &[("If-None-Match", &etag)]).status,
        304
    );
    let started = Instant::now();
    let reply = send(&server, "GET", MODEL_TARGET, &[]);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the client is not left waiting"
    );
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("ETag"), Some(etag.as_str()));
    assert!(reply.broken, "the framing breaks after the kept bytes");
    assert_eq!(reply.body, bytes[..300]);
    assert_eq!(server.requests().last().unwrap().body_bytes, 300);
    // The resume (§1.5): the rest by range, and the next response is whole again.
    let rest = send(
        &server,
        "GET",
        MODEL_TARGET,
        &[("Range", "bytes=300-"), ("If-Range", &etag)],
    );
    assert_eq!(rest.status, 206);
    assert_eq!([reply.body, rest.body].concat(), bytes);
    assert_eq!(send(&server, "GET", MODEL_TARGET, &[]).body, bytes);
}

#[test]
fn truncate_next_applies_to_a_ranged_response_too() {
    let server = serve_fixtures();
    let bytes = fixture(MODEL);
    server.truncate_next(5);
    let reply = send(&server, "GET", MODEL_TARGET, &[("Range", "bytes=100-")]);
    assert_eq!(reply.status, 206);
    assert!(reply.broken);
    assert_eq!(reply.body, bytes[100..105]);
    assert_eq!(
        reply.header("Content-Range"),
        Some(format!("bytes 100-{}/{}", bytes.len() - 1, bytes.len()).as_str())
    );
}

#[test]
fn serve_listing_chooses_the_body_of_the_listing_base() {
    let server = serve_fixtures();
    let base = "/listing/models/";
    let index = "/listing/models/index.json";
    // The default: the player's index.json, the base itself 404 (D-31's first choice).
    assert_eq!(send(&server, "GET", base, &[]).status, 404);
    let reply = send(&server, "GET", index, &[]);
    assert_eq!(
        (reply.status, &reply.body),
        (200, &fixture("listing/index.json"))
    );
    assert_eq!(reply.header("Content-Type"), Some("application/json"));
    let cases = [
        (
            ServedListing::NginxJson,
            "listing/nginx.json",
            "application/json",
        ),
        (
            ServedListing::Html,
            "listing/apache.html",
            "text/html; charset=utf-8",
        ),
    ];
    for (kind, body, content_type) in cases {
        server.serve_listing(kind);
        let reply = send(&server, "GET", base, &[]);
        assert_eq!(
            (reply.status, &reply.body),
            (200, &fixture(body)),
            "{kind:?}"
        );
        assert_eq!(reply.header("Content-Type"), Some(content_type));
        assert_eq!(send(&server, "GET", index, &[]).status, 404, "{kind:?}");
    }
    server.serve_listing(ServedListing::Index);
    assert_eq!(send(&server, "GET", index, &[]).status, 200);
    assert_eq!(send(&server, "GET", base, &[]).status, 404);
}

#[test]
fn the_listed_entries_resolve_under_the_listing_base() {
    let server = serve_fixtures();
    for (target, path) in [
        (
            "/listing/models/snapshot.ply",
            "listing/models/snapshot.ply",
        ),
        (
            "/listing/models/tiny/manifest.json",
            "listing/models/tiny/manifest.json",
        ),
        ("/listing/models/notes.txt", "listing/models/notes.txt"),
    ] {
        assert_eq!(
            send(&server, "GET", target, &[]).body,
            fixture(path),
            "{target}"
        );
    }
    // Listed by the nginx and Apache bodies, absent on disk.
    assert_eq!(
        send(&server, "GET", "/listing/models/.hidden.safetensors", &[]).status,
        404
    );
}

#[test]
fn directories_escapes_and_other_methods_are_refused() {
    let server = serve_fixtures();
    for target in [
        "/",
        "/probes/",
        "/listing/models/tiny/",
        "/listing/models",
        "/../Cargo.toml",
        "/probes/../README.md",
        "/probes/%2e%2e/README.md",
        "/probes//P1.request.json",
        "/bad%zzescape",
    ] {
        assert_eq!(send(&server, "GET", target, &[]).status, 404, "{target}");
    }
    let reply = send(&server, "POST", MODEL_TARGET, &[("Content-Length", "0")]);
    assert_eq!(reply.status, 405);
    assert_eq!(reply.header("Allow"), Some("GET, HEAD"));
}

#[test]
fn servers_are_independent() {
    let a = serve_fixtures();
    let b = serve_fixtures();
    assert_ne!(a.url(), b.url());
    a.serve_listing(ServedListing::Html);
    send(&a, "GET", "/listing/models/", &[]);
    assert_eq!(send(&b, "GET", "/listing/models/", &[]).status, 404);
    assert_eq!((a.requests().len(), b.requests().len()), (1, 1));
}
