// SPDX-License-Identifier: Apache-2.0

//! U-31: `ftgsxr_io::sources` (§1.5 Library and Sources, D-31, D-49; `cpu` lane): `LocalDir`
//! lists one level with kinds; `info` reads the header only; `HttpIndex` with the three
//! flavors over the local server; `DocumentSource` over a stub provider; `open_ref` for every
//! `ModelRef`. D-50 and D-51 (`ModelBytes`; checkpoints travel by path, each route's
//! directory read by `read_checkpoint` equal to the fixture checkpoint, U-5), Q-42 and
//! Q-55 to Q-60 (a checkpoint's cache entry, its pruning, digest, accessor and progress),
//! Q-47 (the module's signatures and rules), Q-48 (`Auto`), over the fixture server of Q-39
//! with the listing choice of Q-6.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ftgsxr_core::{Model, ModelInfo, ModelKind, read_checkpoint, read_exchange};
use ftgsxr_io::{
    DocumentProvider, DocumentSource, Entry, EntryKind, HttpClient, HttpIndex, IoError,
    ListingFlavor, LocalDir, ModelBytes, ModelRef, ModelSource, Progress, SourceSpec, TlsRoots,
    open_ref, open_source,
};
use ftgsxr_testkit::{LocalServer, ServedListing, fixtures_dir, serve_fixtures};
use url::Url;

/// A fresh, empty directory under Cargo's temporary directory for integration tests.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The listing fixture's models directory (tests/fixtures/README.md, Layout).
fn listing_models() -> PathBuf {
    fixtures_dir().join("listing/models")
}

/// A copy of the listing fixture's models, a stray file and a plain subdirectory holding a
/// model one level too deep, in a fresh directory.
fn local_models(name: &str) -> PathBuf {
    let models = listing_models();
    let dir = fresh_dir(name);
    for file in ["probe p1.safetensors", "snapshot.ply", "notes.txt"] {
        fs::copy(models.join(file), dir.join(file)).unwrap();
    }
    fs::create_dir(dir.join("tiny")).unwrap();
    for file in ["manifest.json", "params.safetensors"] {
        fs::copy(models.join("tiny").join(file), dir.join("tiny").join(file)).unwrap();
    }
    fs::create_dir(dir.join("deeper")).unwrap();
    fs::copy(
        models.join("probe p1.safetensors"),
        dir.join("deeper/p1.safetensors"),
    )
    .unwrap();
    dir
}

/// What the test expects of an entry: its name, kind and location.
fn summary(entries: &[Entry]) -> Vec<(String, EntryKind, String)> {
    entries
        .iter()
        .map(|e| (e.name.clone(), e.kind, e.location.clone()))
        .collect()
}

/// The header-only information `read_exchange` agrees with: every field the header gives,
/// read here through the whole-file reader.
fn assert_info_matches_file(info: &ModelInfo, bytes: &[u8]) {
    let model = read_exchange(bytes).expect("the fixture model reads");
    assert_eq!(info.n, model.len() as u64);
    assert_eq!(info.sh_degree, model.sh_degree);
    assert_eq!(info.time, model.time);
    assert_eq!(info.producer, model.producer);
    assert_eq!(info.conventions, model.conventions);
    assert_eq!(info.kind, ModelKind::ExchangeFile);
    assert_eq!(info.file_bytes, bytes.len() as u64);
}

/// `fetch` with the progress reports collected and no cancel.
fn fetch_all(source: &dyn ModelSource, entry: &Entry) -> (ModelRef, ModelBytes, Vec<Progress>) {
    let mut reports = Vec::new();
    let cancel = AtomicBool::new(false);
    let (r, bytes) = source
        .fetch(entry, &mut |p| reports.push(p), &cancel)
        .expect("the entry fetches");
    (r, bytes, reports)
}

#[test]
fn u31_local_dir_lists_one_level_with_kinds() {
    let dir = local_models("u31_local_dir_lists_one_level_with_kinds");
    let source = LocalDir { root: dir.clone() };
    let at = |file: &str| dir.join(file).to_str().unwrap().to_owned();
    let expected = vec![
        (
            "probe p1".to_owned(),
            EntryKind::Exchange,
            at("probe p1.safetensors"),
        ),
        ("snapshot".to_owned(), EntryKind::Ply, at("snapshot.ply")),
        ("tiny".to_owned(), EntryKind::Checkpoint, at("tiny")),
    ];
    assert_eq!(summary(&source.list("").unwrap()), expected);
    assert_eq!(source.name(), dir.to_str().unwrap());

    // The same source through its settings spec (§1.11 `kind` `dir`).
    let spec = SourceSpec {
        kind: "dir".into(),
        location: dir.to_str().unwrap().into(),
        name: "models".into(),
        flavor: None,
    };
    let opened = open_source(&spec, None, None).unwrap();
    assert_eq!(summary(&opened.list("").unwrap()), expected);

    // A subdirectory lists on demand; one outside the root is refused (Q-47).
    let deeper = source.list("deeper").unwrap();
    assert_eq!(
        summary(&deeper),
        vec![(
            "p1".to_owned(),
            EntryKind::Exchange,
            dir.join("deeper")
                .join("p1.safetensors")
                .to_str()
                .unwrap()
                .to_owned()
        )]
    );
    assert_eq!(source.list("../x").unwrap_err().code(), "E-IO");
    assert_eq!(source.list("/").unwrap_err().code(), "E-IO");
}

#[test]
fn u31_local_dir_info_reads_the_header_only() {
    let dir = fresh_dir("u31_local_dir_info_reads_the_header_only");
    let probe = fs::read(listing_models().join("probe p1.safetensors")).unwrap();
    // The same file with every data byte set to 0xFF: the floats become NaN, so the whole
    // file is refused while its header is unchanged.
    let n = u64::from_le_bytes(probe[..8].try_into().unwrap()) as usize;
    let mut garbled = probe.clone();
    garbled[8 + n..].fill(0xFF);
    fs::write(dir.join("p1.safetensors"), &probe).unwrap();
    fs::write(dir.join("garbled.safetensors"), &garbled).unwrap();
    assert!(
        read_exchange(&garbled).is_err(),
        "the garbled file is refused"
    );

    let source = LocalDir { root: dir.clone() };
    let entries = source.list("").unwrap();
    assert_eq!(entries.len(), 2);
    for entry in &entries {
        let info = source.info(entry).expect("the header-only information");
        assert_info_matches_file(&info, &probe);
    }
}

#[test]
fn u31_local_dir_info_reports_a_listed_broken_file() {
    // A broken `ftgs-if` file is listed by its format alone; `info` gives its E-FILE code
    // (Q-38, #84): the probe cut short, and the probe with major version 2.
    let dir = fresh_dir("u31_local_dir_info_reports_a_listed_broken_file");
    let probe = fs::read(listing_models().join("probe p1.safetensors")).unwrap();
    fs::write(dir.join("cut.safetensors"), &probe[..probe.len() - 4]).unwrap();
    let from = br#""version":"1""#;
    let at = probe
        .windows(from.len())
        .position(|w| w == from)
        .expect("the probe's version");
    let mut v2 = probe.clone();
    v2[at + from.len() - 2] = b'2';
    fs::write(dir.join("v2.safetensors"), &v2).unwrap();

    let source = LocalDir { root: dir };
    let entries = source.list("").unwrap();
    let codes: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.name.as_str(), source.info(e).unwrap_err().code()))
        .collect();
    assert_eq!(
        codes,
        vec![("cut", "E-FILE-FORMAT"), ("v2", "E-FILE-VERSION")]
    );
}

#[test]
fn u31_local_dir_info_of_a_kind_without_a_header_read_is_e_io() {
    let dir = local_models("u31_local_dir_info_of_a_kind_without_a_header_read");
    let source = LocalDir { root: dir };
    for entry in source.list("").unwrap() {
        if entry.kind != EntryKind::Exchange {
            assert_eq!(source.info(&entry).unwrap_err().code(), "E-IO", "{entry:?}");
        }
    }
}

#[test]
fn u31_local_dir_fetch_reads_the_file_with_progress() {
    let dir = local_models("u31_local_dir_fetch_reads_the_file_with_progress");
    let source = LocalDir { root: dir.clone() };
    let entries = source.list("").unwrap();
    for entry in entries.iter().filter(|e| e.kind != EntryKind::Checkpoint) {
        let path = PathBuf::from(&entry.location);
        let whole = fs::read(&path).unwrap();
        let (r, bytes, reports) = fetch_all(&source, entry);
        assert_eq!(r, ModelRef::File(path));
        assert_eq!(bytes.as_bytes(), Some(whole.as_slice()));
        let size = whole.len() as u64;
        assert_eq!(
            reports.last(),
            Some(&Progress {
                done: size,
                total: Some(size)
            })
        );
    }
    // A cancelled fetch reads nothing (Q-47).
    let cancel = AtomicBool::new(true);
    let err = source.fetch(&entries[0], &mut |_| {}, &cancel).unwrap_err();
    assert_eq!(err.code(), "E-IO");
    assert!(cancel.load(Ordering::Relaxed));
}

/// The fixture checkpoint read directly from its committed directory (U-5's reader).
fn fixture_checkpoint() -> Model {
    read_checkpoint(&fixtures_dir().join("checkpoint/tiny")).expect("the fixture checkpoint")
}

/// The model a fetched checkpoint names: its directory read by `read_checkpoint` (D-51).
fn read_fetched(bytes: &ModelBytes) -> Model {
    assert_eq!(
        bytes.as_bytes(),
        None,
        "a checkpoint carries no bytes (Q-57)"
    );
    let dir = bytes.checkpoint_dir().expect("a checkpoint directory");
    read_checkpoint(dir).expect("the fetched checkpoint reads")
}

#[test]
fn u31_open_ref_of_a_checkpoint_directory_returns_it() {
    // D-51: a `File` naming a directory is a checkpoint, loaded by path.
    let dir = local_models("u31_open_ref_of_a_checkpoint_directory_returns_it").join("tiny");
    let bytes = open_ref(&ModelRef::File(dir.clone()), None, None).unwrap();
    assert_eq!(bytes, ModelBytes::Checkpoint(dir.clone()));
    assert_eq!(bytes.checkpoint_dir(), Some(dir.as_path()));
    assert_eq!(read_fetched(&bytes), fixture_checkpoint());
}

#[test]
fn u31_local_dir_fetch_of_a_checkpoint_returns_its_directory() {
    // D-51: a local checkpoint loads from its directory; one progress report of the two
    // files' size (Q-38, Q-58).
    let dir = local_models("u31_local_dir_fetch_of_a_checkpoint_returns_its_directory");
    let source = LocalDir { root: dir.clone() };
    let entries = source.list("").unwrap();
    let tiny = entries
        .iter()
        .find(|e| e.kind == EntryKind::Checkpoint)
        .unwrap();
    let (r, bytes, reports) = fetch_all(&source, tiny);
    let tiny_dir = dir.join("tiny");
    assert_eq!(r, ModelRef::File(tiny_dir.clone()));
    assert_eq!(bytes, ModelBytes::Checkpoint(tiny_dir.clone()));
    assert_eq!(read_fetched(&bytes), fixture_checkpoint());
    let size = ["manifest.json", "params.safetensors"]
        .iter()
        .map(|f| fs::metadata(tiny_dir.join(f)).unwrap().len())
        .sum::<u64>();
    assert_eq!(
        reports,
        vec![Progress {
            done: size,
            total: Some(size)
        }]
    );
    // A cancelled fetch is E-IO, as for a file (Q-47).
    let cancel = AtomicBool::new(true);
    assert_eq!(
        source.fetch(tiny, &mut |_| {}, &cancel).unwrap_err().code(),
        "E-IO"
    );
}

/// A client over a fresh cache directory for one test.
fn client(test: &str) -> Arc<HttpClient> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("u31-cache-{test}"));
    let _ = fs::remove_dir_all(&dir);
    Arc::new(HttpClient::new(dir, TlsRoots::WebPki, 1 << 40).expect("a client"))
}

/// The listing base the committed bodies describe (Q-39).
fn listing_base(server: &LocalServer) -> Url {
    Url::parse(&server.url_of("listing/models/")).unwrap()
}

/// The served body each explicit flavor reads (Q-6: the test maps them itself).
const FLAVORS: [(ListingFlavor, ServedListing); 3] = [
    (ListingFlavor::Index, ServedListing::Index),
    (ListingFlavor::NginxJson, ServedListing::NginxJson),
    (ListingFlavor::Html, ServedListing::Html),
];

/// The entries each listing of `listing/models/` gives (§1.5, D-31; names, kinds and
/// locations per Q-36): the index names the checkpoint, the other two list `tiny/` as a
/// directory; dot-names, `notes.txt`, the parent and column-sort links are skipped.
fn expected_entries(base: &Url, served: ServedListing) -> Vec<(String, EntryKind, String)> {
    let at = |path: &str| format!("{base}{path}");
    let probe = (
        "probe p1".to_owned(),
        EntryKind::Exchange,
        at("probe%20p1.safetensors"),
    );
    let snapshot = ("snapshot".to_owned(), EntryKind::Ply, at("snapshot.ply"));
    match served {
        ServedListing::Index => vec![
            probe,
            snapshot,
            ("tiny".to_owned(), EntryKind::Checkpoint, at("tiny/")),
        ],
        ServedListing::NginxJson => vec![
            ("tiny".to_owned(), EntryKind::Dir, at("tiny/")),
            probe,
            snapshot,
        ],
        ServedListing::Html => vec![
            probe,
            snapshot,
            ("tiny".to_owned(), EntryKind::Dir, at("tiny/")),
        ],
    }
}

/// The request targets the server saw, in order.
fn paths(server: &LocalServer) -> Vec<String> {
    server.requests().into_iter().map(|r| r.path).collect()
}

#[test]
fn u31_http_index_lists_each_flavor_explicitly() {
    let server = serve_fixtures();
    let base = listing_base(&server);
    let http = client("explicit");
    for (flavor, served) in FLAVORS {
        server.serve_listing(served);
        let source = HttpIndex::new(base.clone(), flavor, http.clone());
        assert_eq!(source.name(), base.as_str());
        let entries = source
            .list("")
            .unwrap_or_else(|e| panic!("{flavor:?}: {e}"));
        assert_eq!(
            summary(&entries),
            expected_entries(&base, served),
            "{flavor:?}"
        );
    }
    // One request per listing: the index at `<base>index.json`, the other two at the base.
    let base_path = "/listing/models/";
    assert_eq!(
        paths(&server),
        vec![
            format!("{base_path}index.json"),
            base_path.to_owned(),
            base_path.to_owned()
        ]
    );
    // An explicit flavor whose body is not served fails as its request does (Q-48).
    server.serve_listing(ServedListing::Html);
    let index = HttpIndex::new(base.clone(), ListingFlavor::Index, http.clone());
    assert_eq!(index.list("").unwrap_err().code(), "E-NET-FETCH");
    let nginx = HttpIndex::new(base.clone(), ListingFlavor::NginxJson, http);
    assert_eq!(nginx.list("").unwrap_err().code(), "E-NET-LISTING");
}

#[test]
fn u31_http_index_auto_detects_in_d31_order() {
    let server = serve_fixtures();
    let base = listing_base(&server);
    let http = client("auto");
    let base_path = "/listing/models/";
    let index_path = format!("{base_path}index.json");
    for (_, served) in FLAVORS {
        server.serve_listing(served);
        let before = server.requests().len();
        let spec = SourceSpec {
            kind: "http".into(),
            location: base.to_string(),
            name: "fixtures".into(),
            flavor: None,
        };
        let source = open_source(&spec, Some(http.clone()), None).unwrap();
        let entries = source
            .list("")
            .unwrap_or_else(|e| panic!("{served:?}: {e}"));
        assert_eq!(
            summary(&entries),
            expected_entries(&base, served),
            "{served:?}"
        );
        // D-31's order: `index.json` first; the base only when there is none, and once.
        let seen: Vec<String> = paths(&server)[before..].to_vec();
        let expected = match served {
            ServedListing::Index => vec![index_path.clone()],
            _ => vec![index_path.clone(), base_path.to_owned()],
        };
        assert_eq!(seen, expected, "{served:?}");
    }
}

#[test]
fn u31_http_index_auto_without_a_listing_is_e_net_listing() {
    let server = serve_fixtures();
    let http = client("no-listing");
    server.serve_listing(ServedListing::NginxJson);
    // A base answering a body that is neither nginx JSON nor HTML anchors (Q-48).
    let notes = Url::parse(&server.url_of("listing/models/notes.txt")).unwrap();
    let source = HttpIndex::new(notes, ListingFlavor::Auto, http.clone());
    assert_eq!(source.list("").unwrap_err().code(), "E-NET-LISTING");
    // A base that answers 404 (a directory the server lists nowhere, Q-39).
    let probes = Url::parse(&server.url_of("probes/")).unwrap();
    let source = HttpIndex::new(probes, ListingFlavor::Auto, http.clone());
    assert_eq!(source.list("").unwrap_err().code(), "E-NET-LISTING");
    // A server that cannot be reached is E-NET-FETCH (port 1: nothing listens there).
    let gone = Url::parse("http://127.0.0.1:1/models/").unwrap();
    let source = HttpIndex::new(gone, ListingFlavor::Auto, http);
    assert_eq!(source.list("").unwrap_err().code(), "E-NET-FETCH");
}

#[test]
fn u31_http_index_lists_a_subdirectory_on_demand() {
    let server = serve_fixtures();
    let base = listing_base(&server);
    server.serve_listing(ServedListing::NginxJson);
    let source = HttpIndex::new(base.clone(), ListingFlavor::NginxJson, client("subdir"));
    let tiny = source
        .list("")
        .unwrap()
        .into_iter()
        .find(|e| e.kind == EntryKind::Dir)
        .unwrap();
    // The fixture server lists only the base, so the subdirectory's request is a 404; the
    // request shows where the listing was looked for.
    assert!(source.list(&tiny.location).is_err());
    assert_eq!(
        paths(&server).last().map(String::as_str),
        Some("/listing/models/tiny/")
    );
    assert!(source.list("tiny/").is_err());
    assert_eq!(
        paths(&server).last().map(String::as_str),
        Some("/listing/models/tiny/")
    );
    // Outside the base is refused before any request.
    let count = server.requests().len();
    assert_eq!(source.list("../").unwrap_err().code(), "E-IO");
    assert_eq!(server.requests().len(), count);
}

#[test]
fn u31_http_index_info_reads_8_plus_header_length_bytes() {
    let server = serve_fixtures();
    let base = listing_base(&server);
    let http = client("info");
    let whole = fs::read(listing_models().join("probe p1.safetensors")).unwrap();
    let n = u64::from_le_bytes(whole[..8].try_into().unwrap());
    // The index gives the size, the HTML listing does not (Q-36): both read the same bytes.
    for (flavor, served) in [FLAVORS[0], FLAVORS[2]] {
        server.serve_listing(served);
        let source = HttpIndex::new(base.clone(), flavor, http.clone());
        let entries = source.list("").unwrap();
        let probe = entries.iter().find(|e| e.name == "probe p1").unwrap();
        let before = server.requests().len();
        let info = source.info(probe).expect("the header-only information");
        assert_info_matches_file(&info, &whole);
        let records = server.requests()[before..].to_vec();
        assert_eq!(records.len(), 2, "{flavor:?}: the prefix, then the header");
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
}

#[test]
fn u31_http_index_fetch_downloads_and_checks_the_index_digest() {
    let server = serve_fixtures();
    let base = listing_base(&server);
    let source = HttpIndex::new(base.clone(), ListingFlavor::Index, client("fetch"));
    let entries = source.list("").unwrap();
    for entry in entries.iter().filter(|e| e.kind != EntryKind::Checkpoint) {
        assert!(entry.sha256.is_some(), "the index gives the digest");
        let file = if entry.kind == EntryKind::Ply {
            "snapshot.ply"
        } else {
            "probe p1.safetensors"
        };
        let whole = fs::read(listing_models().join(file)).unwrap();
        let (r, bytes, reports) = fetch_all(&source, entry);
        assert_eq!(r, ModelRef::Http(Url::parse(&entry.location).unwrap()));
        assert_eq!(bytes.as_bytes(), Some(whole.as_slice()));
        assert_eq!(reports.last().map(|p| p.done), Some(whole.len() as u64));
    }
    // A digest that disagrees is E-NET-CHECKSUM (§1.5).
    let mut wrong = entries[0].clone();
    wrong.sha256 = Some("0".repeat(64));
    let cancel = AtomicBool::new(false);
    let err = source.fetch(&wrong, &mut |_| {}, &cancel).unwrap_err();
    assert_eq!(err.code(), "E-NET-CHECKSUM");
}

/// The cache directory `client(test)` uses.
fn cache_of(test: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("u31-cache-{test}"))
}

/// The index's checkpoint entry `tiny` over the fixture server.
fn http_checkpoint(source: &HttpIndex) -> Entry {
    source
        .list("")
        .unwrap()
        .into_iter()
        .find(|e| e.kind == EntryKind::Checkpoint)
        .expect("the index names the checkpoint")
}

#[test]
fn u31_http_index_fetch_of_a_checkpoint_fills_one_cache_entry() {
    // D-51: `manifest.json` and `params.safetensors` download into one cache entry, keyed by
    // the directory's URL and holding each under its own name (Q-42), which `fetch` returns
    // as `File(<entry dir>)` and `Checkpoint(<entry dir>)`.
    let server = serve_fixtures();
    let base = listing_base(&server);
    let source = HttpIndex::new(base.clone(), ListingFlavor::Index, client("checkpoint"));
    let tiny = http_checkpoint(&source);
    let before = server.requests().len();
    let (r, bytes, reports) = fetch_all(&source, &tiny);
    let dir = bytes.checkpoint_dir().expect("a checkpoint").to_path_buf();
    assert_eq!(r, ModelRef::File(dir.clone()));
    assert_eq!(read_fetched(&bytes), fixture_checkpoint());
    // The entry of the directory URL, trailing slash included (Q-42).
    let key = ftgsxr_core::hex(&ftgsxr_core::sha256(format!("{base}tiny/").as_bytes()));
    assert_eq!(
        dir.canonicalize().unwrap(),
        cache_of("checkpoint")
            .join("http")
            .join(key)
            .canonicalize()
            .unwrap()
    );
    for name in ["manifest.json", "params.safetensors"] {
        let served = fs::read(listing_models().join("tiny").join(name)).unwrap();
        assert_eq!(fs::read(dir.join(name)).unwrap(), served, "{name}");
        assert!(dir.join(format!("{name}.meta.json")).is_file(), "{name}");
    }
    // The manifest first, then the params, each once.
    let base_path = "/listing/models/tiny/";
    assert_eq!(
        paths(&server)[before..].to_vec(),
        vec![
            format!("{base_path}manifest.json"),
            format!("{base_path}params.safetensors")
        ]
    );
    // Progress is cumulative over both files and ends at their sum (Q-58).
    let size = tiny.size.expect("the index gives the checkpoint's size");
    assert_eq!(
        reports.last(),
        Some(&Progress {
            done: size,
            total: Some(size)
        })
    );
    assert!(reports.windows(2).all(|w| w[0].done <= w[1].done));

    // A second fetch revalidates each file: two 304s, the same entry.
    let before = server.requests().len();
    let (again, bytes, reports) = fetch_all(&source, &tiny);
    assert_eq!(again, ModelRef::File(dir.clone()));
    assert_eq!(read_fetched(&bytes), fixture_checkpoint());
    let statuses: Vec<u16> = server.requests()[before..]
        .iter()
        .map(|r| r.status)
        .collect();
    assert_eq!(statuses, vec![304, 304]);
    assert_eq!(reports.last().map(|p| p.done), Some(size));
}

/// A stub of the Android document provider (§2.5, D-49): one tree, `content://stub/tree`,
/// whose documents are the listing fixture's two model files; every call is recorded.
struct StubProvider {
    calls: Mutex<Vec<String>>,
}

const TREE: &str = "content://stub/tree";
const PROBE_URI: &str = "content://stub/doc/probe";
const PLY_URI: &str = "content://stub/doc/snapshot";

impl StubProvider {
    fn new() -> Arc<StubProvider> {
        Arc::new(StubProvider {
            calls: Mutex::new(Vec::new()),
        })
    }

    fn file_of(uri: &str) -> Option<PathBuf> {
        match uri {
            PROBE_URI => Some(listing_models().join("probe p1.safetensors")),
            PLY_URI => Some(listing_models().join("snapshot.ply")),
            _ => None,
        }
    }

    fn entries() -> Vec<Entry> {
        let entry = |name: &str, uri: &str, kind| Entry {
            name: name.to_owned(),
            location: uri.to_owned(),
            kind,
            size: Some(
                fs::metadata(StubProvider::file_of(uri).unwrap())
                    .unwrap()
                    .len(),
            ),
            mtime: None,
            sha256: None,
            thumbnail: None,
            audio: None,
            scene: None,
        };
        vec![
            entry("probe p1", PROBE_URI, EntryKind::Exchange),
            entry("snapshot", PLY_URI, EntryKind::Ply),
        ]
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl DocumentProvider for StubProvider {
    fn list_tree(&self, tree_uri: &str) -> Result<Vec<Entry>, IoError> {
        self.calls.lock().unwrap().push(format!("list {tree_uri}"));
        if tree_uri == TREE {
            Ok(StubProvider::entries())
        } else {
            Err(IoError::Io(std::io::ErrorKind::NotFound.into()))
        }
    }

    fn open(&self, uri: &str) -> Result<fs::File, IoError> {
        self.calls.lock().unwrap().push(format!("open {uri}"));
        let path = StubProvider::file_of(uri)
            .ok_or_else(|| IoError::Io(std::io::ErrorKind::NotFound.into()))?;
        Ok(fs::File::open(path)?)
    }
}

#[test]
fn u31_document_source_over_a_stub_provider() {
    let stub = StubProvider::new();
    let provider: Arc<dyn DocumentProvider> = stub.clone();
    let spec = SourceSpec {
        kind: "documents".into(),
        location: TREE.into(),
        name: "picked".into(),
        flavor: None,
    };
    let source = open_source(&spec, None, Some(provider.clone())).unwrap();
    assert_eq!(source.name(), TREE);
    let direct = DocumentSource::new(TREE.into(), provider);
    assert_eq!(direct.tree_uri, TREE);

    // The provider's entries, as it gives them (Q-47).
    let entries = source.list("").unwrap();
    assert_eq!(entries, StubProvider::entries());
    assert_eq!(direct.list("").unwrap(), entries);
    assert_eq!(
        stub.calls(),
        vec![format!("list {TREE}"), format!("list {TREE}")]
    );

    let whole = fs::read(listing_models().join("probe p1.safetensors")).unwrap();
    let info = source
        .info(&entries[0])
        .expect("the header-only information");
    assert_info_matches_file(&info, &whole);
    assert_eq!(source.info(&entries[1]).unwrap_err().code(), "E-IO");

    for entry in &entries {
        let file = StubProvider::file_of(&entry.location).unwrap();
        let (r, bytes, reports) = fetch_all(source.as_ref(), entry);
        assert_eq!(r, ModelRef::Document(entry.location.clone()));
        assert_eq!(bytes.as_bytes(), Some(fs::read(file).unwrap().as_slice()));
        assert_eq!(
            reports.last().map(|p| p.done),
            bytes.as_bytes().map(|b| b.len() as u64)
        );
    }
    // A subdirectory is a tree URI the provider is asked for (Q-47).
    assert!(source.list("content://stub/other").is_err());
    assert_eq!(
        stub.calls().last().map(String::as_str),
        Some("list content://stub/other")
    );
}

#[test]
fn u31_open_ref_reads_every_model_ref() {
    let server = serve_fixtures();
    let http = client("open-ref");
    let provider = StubProvider::new();
    let whole = fs::read(listing_models().join("probe p1.safetensors")).unwrap();

    let file = ModelRef::File(listing_models().join("probe p1.safetensors"));
    let url = Url::parse(&server.url_of("listing/models/probe p1.safetensors")).unwrap();
    let web = ModelRef::Http(url);
    let document = ModelRef::Document(PROBE_URI.into());
    for r in [&file, &web, &document] {
        let bytes = open_ref(
            r,
            Some(&http),
            Some(provider.as_ref() as &dyn DocumentProvider),
        )
        .unwrap_or_else(|e| panic!("{r:?}: {e}"));
        assert_eq!(bytes, ModelBytes::Owned(whole.clone()), "{r:?}");
    }
    // Each reference needs only its own reader.
    assert_eq!(
        open_ref(&file, None, None).unwrap().as_bytes(),
        Some(whole.as_slice())
    );
    assert_eq!(open_ref(&web, None, None).unwrap_err().code(), "E-IO");
    assert_eq!(
        open_ref(&document, Some(&http), None).unwrap_err().code(),
        "E-IO"
    );
    // A missing file is E-IO; an unreachable URL E-NET-FETCH.
    let missing = ModelRef::File(listing_models().join("absent.safetensors"));
    assert_eq!(open_ref(&missing, None, None).unwrap_err().code(), "E-IO");
    let absent = Url::parse(&server.url_of("listing/models/absent.safetensors")).unwrap();
    assert_eq!(
        open_ref(&ModelRef::Http(absent), Some(&http), None)
            .unwrap_err()
            .code(),
        "E-NET-FETCH"
    );
}

#[test]
fn u31_open_source_refuses_what_it_cannot_serve() {
    let spec = |kind: &str, location: &str| SourceSpec {
        kind: kind.into(),
        location: location.into(),
        name: String::new(),
        flavor: Some(ListingFlavor::Html),
    };
    let code = |s: &SourceSpec, client: Option<Arc<HttpClient>>| {
        open_source(s, client, None).err().map(|e| e.code())
    };
    // §1.11 spells the kinds exactly (Q-47).
    assert_eq!(code(&spec("ftp", "x"), None), Some("E-IO"));
    assert_eq!(code(&spec("Dir", "x"), None), Some("E-IO"));
    assert_eq!(code(&spec("http", "http://127.0.0.1/"), None), Some("E-IO"));
    assert_eq!(code(&spec("documents", TREE), None), Some("E-IO"));
    let http = client("open-source");
    assert_eq!(
        code(&spec("http", "not a url"), Some(http.clone())),
        Some("E-NET-FETCH")
    );
    assert_eq!(code(&spec("http", "http://127.0.0.1/m/"), Some(http)), None);
}
