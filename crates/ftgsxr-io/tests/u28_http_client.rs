// SPDX-License-Identifier: Apache-2.0

//! U-28: `ftgsxr_io::http::HttpClient` (`get_text`, `get_range`, `download`, `prune`) against
//! `ftgsxr_testkit::serve_fixtures` (§1.5 Download cache, §1.11 Cache entries, D-20; Q-4, Q-5,
//! Q-T18.1 to Q-T18.4, the fixture server of Q-39 and its truncation Q-40).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ftgsxr_io::http::{HttpClient, Progress, TlsRoots};
use ftgsxr_testkit::{fixtures_dir, serve_fixtures};
use url::Url;

/// A fresh, empty cache directory for one test, named after it so tests sharing the process
/// never meet and a rerun replaces the last run's.
fn cache_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ftgsxr-u28-{test}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a temporary cache directory");
    dir
}

/// A subscriber that counts the WARN events it sees and nothing else.
struct WarnCounter(Arc<AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The number of warnings `f` logs on this thread.
fn warnings_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let count = Arc::new(AtomicUsize::new(0));
    let value = tracing::subscriber::with_default(WarnCounter(count.clone()), f);
    (value, count.load(Ordering::SeqCst))
}

#[test]
fn u28_both_tls_roots_construct_and_platform_warns() {
    let dir = cache_dir("tls");
    let (webpki, warned) =
        warnings_during(|| HttpClient::new(dir.clone(), TlsRoots::WebPki, 1 << 30));
    assert!(webpki.is_ok(), "WebPki roots construct a client");
    assert_eq!(warned, 0, "the WebPki roots log no warning");
    let (platform, warned) =
        warnings_during(|| HttpClient::new(dir.clone(), TlsRoots::Platform, 1 << 30));
    assert!(platform.is_ok(), "Platform roots construct a client (Q-5)");
    assert_eq!(warned, 1, "the Platform fallback logs one warning (Q-5)");
}

/// A client over a fresh cache directory with a generous size limit.
fn client(test: &str) -> HttpClient {
    HttpClient::new(cache_dir(test), TlsRoots::WebPki, 1 << 40).expect("a client")
}

/// The fixture server's URL of `path` as a `Url`.
fn url_of(server: &ftgsxr_testkit::LocalServer, path: &str) -> Url {
    Url::parse(&server.url_of(path)).expect("a fixture URL")
}

#[test]
fn u28_header_only_read_fetches_8_plus_header_length_bytes() {
    let server = serve_fixtures();
    let http = client("header");
    let url = url_of(&server, "checkpoint/tiny.safetensors");
    let whole = std::fs::read(fixtures_dir().join("checkpoint/tiny.safetensors")).unwrap();

    let prefix = http.get_range(&url, 0, 8).expect("the length prefix");
    assert_eq!(prefix, whole[..8]);
    let n = u64::from_le_bytes(prefix.try_into().unwrap());
    let header = http.get_range(&url, 8, n).expect("the JSON header");
    assert_eq!(header, whole[8..8 + n as usize]);

    let records = server.requests();
    assert_eq!(records.len(), 2, "two requests, no whole-file read");
    assert_eq!(records[0].header("Range"), Some("bytes=0-7"));
    assert_eq!(
        records[1].header("Range"),
        Some(format!("bytes=8-{}", 8 + n - 1).as_str())
    );
    assert!(records.iter().all(|r| r.status == 206));
    let fetched: u64 = records.iter().map(|r| r.body_bytes).sum();
    assert_eq!(fetched, 8 + n, "8 + header-length bytes on the wire");
    assert!(fetched < whole.len() as u64);
}

#[test]
fn u28_get_text_reads_a_listing_body_and_refuses_a_missing_one() {
    let server = serve_fixtures();
    let http = client("text");
    let text = http
        .get_text(&url_of(&server, "listing/models/index.json"))
        .expect("the served index");
    let committed = std::fs::read_to_string(fixtures_dir().join("listing/index.json")).unwrap();
    assert_eq!(text, committed);
    let missing = http.get_text(&url_of(&server, "listing/absent.json"));
    assert_eq!(missing.map_err(|e| e.code()), Err("E-NET-FETCH"));
}

/// The download cache entry of `url` (§1.11: `http/<sha256 of the URL>/`).
fn entry_dir(cache: &std::path::Path, url: &Url) -> PathBuf {
    let digest = ftgsxr_core::sha256(url.as_str().as_bytes());
    cache.join("http").join(ftgsxr_core::hex(&digest))
}

/// `download` with no expected digest, no cancel and the progress reports collected.
fn fetch(http: &HttpClient, url: &Url) -> (Result<PathBuf, ftgsxr_io::IoError>, Vec<Progress>) {
    let mut reports = Vec::new();
    let cancel = AtomicBool::new(false);
    let result = http.download(url, None, &mut |p| reports.push(p), &cancel);
    (result, reports)
}

/// The quoted SHA-256 the fixture server gives as a file's ETag (Q-39).
fn etag_of(bytes: &[u8]) -> String {
    format!("\"{}\"", ftgsxr_core::hex(&ftgsxr_core::sha256(bytes)))
}

#[test]
fn u28_a_304_on_if_none_match_reuses_the_cached_file() {
    let server = serve_fixtures();
    let cache = cache_dir("revalidate");
    let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, 1 << 40).unwrap();
    let url = url_of(&server, "probes/P1.renders.safetensors");
    let whole = std::fs::read(fixtures_dir().join("probes/P1.renders.safetensors")).unwrap();

    let (first, reports) = fetch(&http, &url);
    let path = first.expect("the first download");
    assert_eq!(
        path,
        entry_dir(&cache, &url).join("file"),
        "§1.11's cache entry"
    );
    assert_eq!(std::fs::read(&path).unwrap(), whole);
    let size = whole.len() as u64;
    assert_eq!(
        reports.last(),
        Some(&Progress {
            done: size,
            total: Some(size)
        })
    );

    let (second, _) = fetch(&http, &url);
    assert_eq!(second.expect("the revalidated download"), path);
    assert_eq!(std::fs::read(&path).unwrap(), whole);
    let records = server.requests();
    assert_eq!(records.len(), 2);
    assert_eq!(
        records[1].header("If-None-Match"),
        Some(etag_of(&whole).as_str())
    );
    assert!(
        records[1].header("If-Modified-Since").is_some(),
        "both validators sent (§1.5)"
    );
    assert_eq!((records[1].status, records[1].body_bytes), (304, 0));
}

#[test]
fn u28_a_resume_after_a_truncated_download_gives_the_whole_file() {
    let server = serve_fixtures();
    let cache = cache_dir("resume");
    let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, 1 << 40).unwrap();
    let url = url_of(&server, "probes/P2.renders.safetensors");
    let whole = std::fs::read(fixtures_dir().join("probes/P2.renders.safetensors")).unwrap();
    let keep = 300_000u64;

    server.truncate_next(keep);
    let (cut, _) = fetch(&http, &url);
    assert_eq!(cut.map_err(|e| e.code()), Err("E-NET-FETCH"));
    assert!(
        !entry_dir(&cache, &url).join("file").exists(),
        "no complete file yet"
    );

    let (resumed, reports) = fetch(&http, &url);
    let path = resumed.expect("the resumed download");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        whole,
        "bytes identical to the whole file"
    );
    assert!(
        reports.iter().all(|p| p.done > keep),
        "progress counts the bytes already held"
    );

    let records = server.requests();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].body_bytes, keep);
    assert_eq!(
        records[1].header("Range"),
        Some(format!("bytes={keep}-").as_str())
    );
    assert_eq!(
        records[1].header("If-Range"),
        Some(etag_of(&whole).as_str())
    );
    assert_eq!(records[1].status, 206);
    assert_eq!(
        records[1].body_bytes,
        whole.len() as u64 - keep,
        "only the rest is fetched"
    );
}

#[test]
fn u28_a_checksum_mismatch_discards_the_download() {
    let server = serve_fixtures();
    let cache = cache_dir("checksum");
    let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, 1 << 40).unwrap();
    let url = url_of(&server, "probes/P3.renders.safetensors");
    let whole = std::fs::read(fixtures_dir().join("probes/P3.renders.safetensors")).unwrap();
    let cancel = AtomicBool::new(false);
    let wrong = "0".repeat(64);

    let result = http.download(&url, Some(&wrong), &mut |_| {}, &cancel);
    assert_eq!(result.map_err(|e| e.code()), Err("E-NET-CHECKSUM"));
    let entry = entry_dir(&cache, &url);
    let left: Vec<_> = std::fs::read_dir(&entry)
        .map(|dir| dir.map(|e| e.unwrap().file_name()).collect())
        .unwrap_or_default();
    assert!(
        left.is_empty(),
        "the download is discarded, nothing kept: {left:?}"
    );

    let right = ftgsxr_core::hex(&ftgsxr_core::sha256(&whole)).to_uppercase();
    let path = http
        .download(&url, Some(&right), &mut |_| {}, &cancel)
        .expect("the matching digest, in either case");
    assert_eq!(std::fs::read(path).unwrap(), whole);
}

#[test]
fn u28_cancel_stops_within_one_chunk() {
    let server = serve_fixtures();
    let cache = cache_dir("cancel");
    let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, 1 << 40).unwrap();
    let url = url_of(&server, "probes/P4.renders.safetensors");
    let whole = std::fs::read(fixtures_dir().join("probes/P4.renders.safetensors")).unwrap();

    // Cancelled before it starts: no request at all.
    let cancel = AtomicBool::new(true);
    let early = http.download(&url, None, &mut |_| {}, &cancel);
    assert_eq!(early.map_err(|e| e.code()), Err("E-NET-FETCH"));
    assert!(server.requests().is_empty());

    // Cancelled at the first chunk's report: no further chunk is read or reported.
    let cancel = AtomicBool::new(false);
    let mut reports = Vec::new();
    let result = http.download(
        &url,
        None,
        &mut |p| {
            reports.push(p);
            cancel.store(true, Ordering::SeqCst);
        },
        &cancel,
    );
    assert_eq!(result.map_err(|e| e.code()), Err("E-NET-FETCH"));
    assert_eq!(
        reports.len(),
        1,
        "the download stops after the chunk it was cancelled in"
    );
    let held = reports[0].done;
    assert!(held > 0 && held < whole.len() as u64);
    let part_bytes: u64 = std::fs::read_dir(entry_dir(&cache, &url))
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| {
            e.file_name() != "meta.json" && !e.file_name().to_string_lossy().ends_with(".json")
        })
        .map(|e| e.metadata().unwrap().len())
        .sum();
    assert_eq!(
        part_bytes, held,
        "the cancelled download keeps exactly what it reported"
    );

    // A later download resumes it to the whole file (Q-T18.2).
    let (resumed, _) = fetch(&http, &url);
    assert_eq!(
        std::fs::read(resumed.expect("the resumed download")).unwrap(),
        whole
    );
    assert_eq!(
        server.requests()[1].header("Range"),
        Some(format!("bytes={held}-").as_str())
    );
}

/// The bytes of every file in a cache entry.
fn entry_bytes(entry: &std::path::Path) -> u64 {
    std::fs::read_dir(entry)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum()
}

#[test]
fn u28_prune_evicts_least_recently_used_and_never_a_file_in_use() {
    let server = serve_fixtures();
    let cache = cache_dir("prune");
    let names = ["P5", "P6", "P7"].map(|p| format!("probes/{p}.renders.safetensors"));
    let urls = names.each_ref().map(|n| url_of(&server, n));
    let size = std::fs::metadata(fixtures_dir().join(&names[0]))
        .unwrap()
        .len();
    // Room for two of the three files and their records, not for three.
    let http = HttpClient::new(cache.clone(), TlsRoots::WebPki, 2 * size + 64 * 1024).unwrap();
    let paths = urls
        .each_ref()
        .map(|u| fetch(&http, u).0.expect("a download"));
    // A is used again (a 304), so B is now the least recently used.
    assert_eq!(fetch(&http, &urls[0]).0.unwrap(), paths[0]);
    let b_bytes = entry_bytes(&entry_dir(&cache, &urls[1]));

    assert_eq!(http.prune().expect("a prune"), b_bytes, "the bytes freed");
    assert!(paths[0].exists() && paths[2].exists());
    assert!(!paths[1].exists(), "the least recently used entry goes");
    assert_eq!(http.prune().unwrap(), 0, "already below max_bytes");

    // Room for one: C is now the least recently used, but it is in use.
    let small = HttpClient::new(cache.clone(), TlsRoots::WebPki, size + 64 * 1024).unwrap();
    small.set_in_use(&[paths[2].clone()]);
    let a_bytes = entry_bytes(&entry_dir(&cache, &urls[0]));
    assert_eq!(small.prune().unwrap(), a_bytes);
    assert!(paths[2].exists(), "a file in use is never evicted (Q-4)");
    assert!(!paths[0].exists());

    // Nothing in use: C goes too once the limit is below it.
    let tiny = HttpClient::new(cache.clone(), TlsRoots::WebPki, 0).unwrap();
    small.set_in_use(&[]);
    tiny.set_in_use(&[paths[2].clone()]);
    assert_eq!(tiny.prune().unwrap(), 0, "in use, whatever the limit");
    tiny.set_in_use(&[]);
    assert!(tiny.prune().unwrap() > 0);
    assert!(!paths[2].exists());
}
