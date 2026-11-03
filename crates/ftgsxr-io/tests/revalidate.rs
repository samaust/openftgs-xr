// SPDX-License-Identifier: Apache-2.0

//! Added for #375 (§1.5 Library's ETag on refresh, Q-229, Q-231): `HttpClient::revalidate`
//! against `ftgsxr_testkit::serve_fixtures`, whose ETag is the file's quoted SHA-256 (Q-39).
//! No §3 row pins it; the source of truth is §1.5's "size, ETag or SHA-256".

use std::path::PathBuf;

use ftgsxr_io::http::{HttpClient, Revalidation, TlsRoots};
use ftgsxr_testkit::{LocalServer, fixtures_dir, serve_fixtures};
use url::Url;

const PROBE: &str = "listing/models/probe p1.safetensors";
const PROBE_TARGET: &str = "/listing/models/probe%20p1.safetensors";

/// A client over a fresh cache directory of its own.
fn client(test: &str) -> HttpClient {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("ftgsxr-revalidate-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    HttpClient::new(dir, TlsRoots::WebPki, 1 << 30).expect("a client")
}

fn probe_url(server: &LocalServer) -> Url {
    Url::parse(&server.url_of(PROBE)).expect("a fixture URL")
}

/// The probe's ETag as the fixture server gives it (Q-39).
fn probe_etag() -> String {
    let bytes = std::fs::read(fixtures_dir().join(PROBE)).expect("the probe");
    format!("\"{}\"", ftgsxr_core::hex(&ftgsxr_core::sha256(&bytes)))
}

#[test]
fn without_an_etag_a_head_gives_the_current_one() {
    let server = serve_fixtures();
    let http = client("none");
    let answer = http
        .revalidate(&probe_url(&server), None)
        .map_err(|e| e.code());
    assert_eq!(answer, Ok(Revalidation::Current(Some(probe_etag()))));
    let records = server.requests();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].method, "HEAD");
    assert_eq!(records[0].path, PROBE_TARGET);
    assert_eq!(records[0].header("If-None-Match"), None);
    assert_eq!(records[0].body_bytes, 0);
}

#[test]
fn the_files_own_etag_is_unchanged_through_a_304() {
    let server = serve_fixtures();
    let http = client("same");
    let etag = probe_etag();
    let answer = http
        .revalidate(&probe_url(&server), Some(&etag))
        .map_err(|e| e.code());
    assert_eq!(answer, Ok(Revalidation::Unchanged));
    let records = server.requests();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].method, "HEAD");
    assert_eq!(records[0].header("If-None-Match"), Some(etag.as_str()));
    assert_eq!(records[0].status, 304);
}

#[test]
fn another_etag_gives_the_files_current_one() {
    let server = serve_fixtures();
    let http = client("other");
    let answer = http
        .revalidate(&probe_url(&server), Some("\"other\""))
        .map_err(|e| e.code());
    assert_eq!(answer, Ok(Revalidation::Current(Some(probe_etag()))));
    assert_eq!(server.requests()[0].status, 200);
}

#[test]
fn a_2xx_without_an_etag_is_current_with_none() {
    let server = serve_fixtures();
    server.status_for(PROBE_TARGET, 200);
    let http = client("bare");
    let answer = http
        .revalidate(&probe_url(&server), Some(&probe_etag()))
        .map_err(|e| e.code());
    assert_eq!(answer, Ok(Revalidation::Current(None)));
}

// Q-231
#[test]
fn an_error_status_or_no_server_is_e_net_fetch() {
    let server = serve_fixtures();
    server.status_for(PROBE_TARGET, 405);
    let http = client("error");
    let url = probe_url(&server);
    let answer = http.revalidate(&url, Some(&probe_etag()));
    assert_eq!(answer.map_err(|e| e.code()), Err("E-NET-FETCH"));
    drop(server);
    // A fresh client: the first one may hold a pooled connection to the stopped server.
    let answer = client("gone").revalidate(&url, None);
    assert_eq!(answer.map_err(|e| e.code()), Err("E-NET-FETCH"));
}
