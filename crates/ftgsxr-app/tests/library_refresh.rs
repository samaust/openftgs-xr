// SPDX-License-Identifier: Apache-2.0

//! Added for #375 (`cpu`): a library refresh keeps an entry's information only while its
//! file is unchanged, the ETag of an HTTP entry included (§1.5 Library: "entries whose file
//! changed (size, ETag or SHA-256) lose their cached information"; Q-230, Q-232). No §3 row
//! pins a refresh; §1.5's sentence is the source of truth.
//!
//! Added for #389: the listing's `mtime` (Q-231) and the two 200 answers to the ETag check,
//! another ETag and the same one (Q-230), served by a server private to these tests.
//!
//! Added for #400: the ETag of the `HEAD` sent before a load (Q-242).

use crate::support;

use bevy::asset::Handle;
use ftgsxr_app::{Library, LibraryCommand, LibraryEntry, LoadedModel};
use ftgsxr_io::{EntryKind, ListingFlavor, ModelRef, SourceSpec, Url};
use ftgsxr_testkit::{RequestRecord, ServedListing, fixtures_dir};
use std::sync::{Arc, Mutex};
use support::TestApp;

/// The probe's path on the server, as sent.
const PROBE: &str = "/listing/models/probe%20p1.safetensors";

/// Adds `spec` as source 1 and waits for the probe's header-only information.
fn add_and_wait(t: &mut TestApp, spec: SourceSpec) {
    t.command(LibraryCommand::AddSource(spec));
    assert!(
        t.until(2000, |app| probe_has_info(
            app.world().resource::<Library>()
        )),
        "the source lists and the probe's information arrives"
    );
}

/// The fixture server's listing base as an HTTP source (`index.json`).
fn http_source(t: &mut TestApp) {
    t.server.serve_listing(ServedListing::Index);
    let location = t.server.url_of("listing/models/");
    add_and_wait(
        t,
        SourceSpec {
            kind: "http".into(),
            location,
            name: "fixtures".into(),
            flavor: Some(ListingFlavor::Index),
        },
    );
}

fn probe_has_info(lib: &Library) -> bool {
    lib.entries
        .iter()
        .any(|e| e.source == 1 && e.entry.kind == EntryKind::Exchange && e.info.is_some())
}

/// The requests for the probe from record `from` on.
fn probe_requests(t: &TestApp, from: usize) -> Vec<RequestRecord> {
    requests_for(t, PROBE, from)
}

/// Advances frames until `done` holds of the app and its server, at most `max` frames.
fn pump_until(t: &mut TestApp, max: usize, done: impl Fn(&TestApp) -> bool) -> bool {
    for _ in 0..max {
        t.app.update();
        if done(t) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    false
}

/// Refreshes source 1 and waits for the probe's `HEAD` after record `from`.
fn refresh_until_head(t: &mut TestApp, from: usize) -> RequestRecord {
    t.command(LibraryCommand::Refresh(1));
    let head = |t: &TestApp| {
        probe_requests(t, from)
            .into_iter()
            .find(|r| r.method == "HEAD")
    };
    pump_until(t, 2000, |t| head(t).is_some());
    head(t).expect("the refresh revalidates the probe with a HEAD")
}

/// Whether the probe was read (a `GET`) after record `from`.
fn read_again(t: &TestApp, from: usize) -> bool {
    probe_requests(t, from).iter().any(|r| r.method == "GET")
}

#[test]
fn an_unchanged_etag_keeps_the_information_without_a_new_read() {
    let mut t = TestApp::new("refresh-same");
    http_source(&mut t);
    let before = t.server.requests().len();
    let head = refresh_until_head(&mut t, before);
    assert!(head.header("If-None-Match").is_some(), "{head:?}");
    assert_eq!(head.status, 304);
    t.until(200, |_| false);
    assert!(probe_has_info(t.app.world().resource::<Library>()));
    assert!(!read_again(&t, before), "no header read after a 304");
}

#[test]
fn another_etag_drops_the_information_and_reads_it_again() {
    let mut t = TestApp::new("refresh-changed");
    http_source(&mut t);
    // The server now answers the probe with a bare 200: its ETag no longer matches.
    t.server.status_for(PROBE, 200);
    let before = t.server.requests().len();
    let head = refresh_until_head(&mut t, before);
    assert_eq!(head.status, 200);
    assert!(
        pump_until(&mut t, 2000, |t| read_again(t, before)),
        "the header is read again"
    );
    pump_until(&mut t, 200, |_| false);
    // The bare 200 has no bytes, so the new read fails and the information stays empty.
    assert!(!probe_has_info(t.app.world().resource::<Library>()));
}

// Q-232
#[test]
fn a_failed_check_keeps_the_information() {
    let mut t = TestApp::new("refresh-failed");
    http_source(&mut t);
    t.server.status_for(PROBE, 500);
    let before = t.server.requests().len();
    let head = refresh_until_head(&mut t, before);
    assert_eq!(head.status, 500);
    t.until(200, |_| false);
    assert!(probe_has_info(t.app.world().resource::<Library>()));
    assert!(
        !read_again(&t, before),
        "no header read after a failed check"
    );
}

// Added for #400 (`cpu`): an HTTP entry whose information a load fills keeps the ETag of a
// `HEAD` sent before the load's download, and a refresh checks it (Q-242's decision cell is
// the source of truth: the location, or a checkpoint's `manifest.json`).

/// The PLY's path on the server, as sent.
const PLY: &str = "/listing/models/snapshot.ply";
/// The listing's digest sidecar (Q-47).
const SUMS: &str = "/listing/models/SHA256SUMS";
/// The checkpoint's manifest's path on the server, as sent.
const MANIFEST: &str = "/listing/models/tiny/manifest.json";

/// Adds the fixture server's listing base in `flavor` as source 1 and waits for an entry
/// of `kind`; that entry's index.
fn listed_source(t: &mut TestApp, flavor: ServedListing, kind: EntryKind) -> usize {
    t.server.serve_listing(flavor);
    let flavor = match flavor {
        ServedListing::Index => ListingFlavor::Index,
        ServedListing::NginxJson => ListingFlavor::NginxJson,
        ServedListing::Html => ListingFlavor::Html,
    };
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: t.server.url_of("listing/models/"),
        name: "fixtures".into(),
        flavor: Some(flavor),
    }));
    assert!(
        t.until(2000, |app| entry_index(
            app.world().resource::<Library>(),
            kind
        )
        .is_some()),
        "the source lists a {kind:?} entry"
    );
    entry_index(t.app.world().resource::<Library>(), kind).unwrap()
}

fn entry_index(lib: &Library, kind: EntryKind) -> Option<usize> {
    lib.entries
        .iter()
        .position(|e| e.source == 1 && e.entry.kind == kind)
}

fn entry(t: &TestApp, i: usize) -> LibraryEntry {
    t.app.world().resource::<Library>().entries[i].clone()
}

/// Loads entry `i` and waits for its identity; gives it a thumbnail, as the UI would, so
/// that its drop is seen (the CPU app renders none).
fn load_until_identified(t: &mut TestApp, i: usize) {
    t.command(LibraryCommand::Load {
        entry: i,
        reduce: false,
    });
    assert!(
        t.until(3000, |app| app.world().resource::<Library>().entries[i]
            .identity
            .is_some()),
        "the load fills the entry's identity: {:?}",
        entry(t, i).state
    );
    let e = &mut t.app.world_mut().resource_mut::<Library>().entries[i];
    assert!(e.info.is_some(), "the load fills the information");
    e.thumbnail = Some(Handle::default());
}

/// The requests for `path` from record `from` on.
fn requests_for(t: &TestApp, path: &str, from: usize) -> Vec<RequestRecord> {
    t.server.requests()[from..]
        .iter()
        .filter(|r| r.path == path)
        .cloned()
        .collect()
}

/// Refreshes source 1 after `path` changed on the server (a bare 200, so another ETag) and
/// checks that the refresh sends a conditional `HEAD` of `path` and drops entry `i`'s
/// information, identity and thumbnail.
fn a_change_of(t: &mut TestApp, i: usize, path: &str) {
    t.server.status_for(path, 200);
    let before = t.server.requests().len();
    t.command(LibraryCommand::Refresh(1));
    let head = |t: &TestApp| {
        requests_for(t, path, before)
            .into_iter()
            .find(|r| r.method == "HEAD")
    };
    pump_until(t, 2000, |t| head(t).is_some());
    let head = head(t).expect("the refresh revalidates the entry with a HEAD");
    assert!(head.header("If-None-Match").is_some(), "{head:?}");
    assert!(
        pump_until(t, 2000, |t| entry(t, i).identity.is_none()),
        "another ETag drops the identity"
    );
    let e = entry(t, i);
    assert_eq!((e.info, e.thumbnail), (None, None));
}

#[test]
fn a_ply_entry_loaded_over_http_and_changed_on_the_server_loses_its_identity() {
    let mut t = TestApp::new("refresh-ply-load");
    // nginx's listing gives no SHA-256, and the file keeps its size and mtime. The listing's
    // `SHA256SUMS` holds a wrong digest for the PLY on purpose (Q-47's fixture), so it is
    // hidden: the download then has no digest either.
    t.server.status_for(SUMS, 404);
    let ply = listed_source(&mut t, ServedListing::NginxJson, EntryKind::Ply);
    assert_eq!(entry(&t, ply).entry.sha256, None);
    load_until_identified(&mut t, ply);
    a_change_of(&mut t, ply, PLY);
}

#[test]
fn a_checkpoint_loaded_over_http_whose_manifest_changed_loses_its_identity() {
    let mut t = TestApp::new("refresh-checkpoint-load");
    // The index gives the checkpoint no SHA-256; its ETag is its `manifest.json`'s.
    let tiny = listed_source(&mut t, ServedListing::Index, EntryKind::Checkpoint);
    assert_eq!(entry(&t, tiny).entry.sha256, None);
    load_until_identified(&mut t, tiny);
    a_change_of(&mut t, tiny, MANIFEST);
}

#[test]
fn the_loads_head_goes_before_its_first_get() {
    let mut t = TestApp::new("refresh-head-first");
    t.server.status_for(SUMS, 404);
    let ply = listed_source(&mut t, ServedListing::NginxJson, EntryKind::Ply);
    let before = t.server.requests().len();
    load_until_identified(&mut t, ply);
    let methods: Vec<String> = requests_for(&t, PLY, before)
        .into_iter()
        .map(|r| r.method)
        .collect();
    assert_eq!(methods, ["HEAD", "GET"], "the PLY's requests, in order");
    // And before the download's digest sidecars (Q-47), which the fetch asks first.
    let first = t.server.requests()[before..]
        .iter()
        .position(|r| r.path == PLY || r.path == SUMS || r.path.ends_with(".sha256"))
        .map(|i| t.server.requests()[before + i].clone());
    assert_eq!(
        first.map(|r| (r.method, r.path)),
        Some(("HEAD".into(), PLY.into()))
    );
}

// Q-232
#[test]
fn a_load_whose_head_fails_still_loads_and_keeps_no_etag() {
    let mut t = TestApp::new("refresh-head-failed");
    // The PLY's download is cached first by a reference load (no library `HEAD`), so that a
    // load from the index, whose digest matches the cached record, sends no `GET` (Q-43)
    // and succeeds while every request for the file answers 500.
    let url = Url::parse(&t.server.url_of("listing/models/snapshot.ply")).unwrap();
    t.command(LibraryCommand::LoadRef(ModelRef::Http(url)));
    assert!(
        t.until(3000, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "the reference loads"
    );
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    t.server.status_for(PLY, 500);
    let ply = listed_source(&mut t, ServedListing::Index, EntryKind::Ply);
    let before = t.server.requests().len();
    load_until_identified(&mut t, ply);
    let heads: Vec<u16> = requests_for(&t, PLY, before)
        .iter()
        .filter(|r| r.method == "HEAD")
        .map(|r| r.status)
        .collect();
    assert_eq!(heads, [500], "the load's HEAD fails");
    let before = t.server.requests().len();
    t.command(LibraryCommand::Refresh(1));
    t.until(300, |_| false);
    assert!(
        requests_for(&t, PLY, before).is_empty(),
        "no check for an entry without an ETag"
    );
    assert!(entry(&t, ply).identity.is_some(), "the entry is unchanged");
}

#[test]
fn an_entry_without_an_etag_is_compared_as_before() {
    // A local directory has no ETag: its unchanged entries keep their information.
    let mut t = TestApp::new("refresh-local");
    let location = fixtures_dir()
        .join("listing/models")
        .to_string_lossy()
        .into_owned();
    add_and_wait(
        &mut t,
        SourceSpec {
            kind: "dir".into(),
            location,
            name: "local".into(),
            flavor: None,
        },
    );
    t.command(LibraryCommand::Refresh(1));
    t.until(200, |_| false);
    assert!(probe_has_info(t.app.world().resource::<Library>()));
    assert!(
        t.server.requests().is_empty(),
        "no request for a local source"
    );
}

/// The probe's file name on the refresh server.
const REFRESH_FILE: &str = "probe.safetensors";

/// What the refresh server answers with, changed by the test between two refreshes.
struct Served {
    /// The probe's `mtime` in the nginx JSON listing.
    mtime: String,
    /// The probe's ETag.
    etag: String,
    /// Whether a `HEAD` whose `If-None-Match` names the ETag is answered 304; when false,
    /// the server ignores `If-None-Match` and answers 200 with the ETag.
    honour_none_match: bool,
}

/// A request the refresh server answered for the probe.
#[derive(Clone, Debug)]
struct Seen {
    method: String,
    none_match: Option<String>,
    status: u16,
}

/// Added for #389: an HTTP server private to these tests, whose answers change between two
/// refreshes, which the testkit's `LocalServer` does not offer (Q-39). It serves nginx's JSON
/// listing of the probe at `/m/` (its size and `mtime`), and the probe at `/m/probe.safetensors`:
/// a `HEAD` gets the probe's ETag (or 304, see `Served`), a `GET` the bytes asked for.
struct RefreshServer {
    base: String,
    served: Arc<Mutex<Served>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl RefreshServer {
    fn start() -> RefreshServer {
        let body = std::fs::read(fixtures_dir().join("listing/models/probe p1.safetensors"))
            .expect("the probe fixture");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let served = Arc::new(Mutex::new(Served {
            mtime: "Thu, 29 Oct 2026 12:00:00 GMT".into(),
            etag: "\"v1\"".into(),
            honour_none_match: true,
        }));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (shared, log, body) = (served.clone(), seen.clone(), Arc::new(body));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { return };
                let (served, log, body) = (shared.clone(), log.clone(), body.clone());
                std::thread::spawn(move || answer(stream, &body, &served, &log));
            }
        });
        RefreshServer {
            base: format!("http://127.0.0.1:{port}/m/"),
            served,
            seen,
        }
    }

    fn change(&self, f: impl FnOnce(&mut Served)) {
        f(&mut self.served.lock().unwrap());
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

/// Answers the requests of one connection (`RefreshServer`).
fn answer(
    mut stream: std::net::TcpStream,
    body: &[u8],
    served: &Mutex<Served>,
    seen: &Mutex<Vec<Seen>>,
) {
    use std::io::{BufRead, BufReader, Write};
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split(' ');
        let method = parts.next().unwrap_or("").to_owned();
        let path = parts.next().unwrap_or("").to_owned();
        let (mut range, mut none_match) = (None, None);
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                break;
            }
            let Some((name, value)) = h.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "range" => {
                    let (a, b) = value
                        .strip_prefix("bytes=")
                        .and_then(|v| v.split_once('-'))
                        .unwrap();
                    let a: usize = a.parse().unwrap();
                    let b = b
                        .parse::<usize>()
                        .map_or(body.len() - 1, |b| b.min(body.len() - 1));
                    range = Some((a, b));
                }
                "if-none-match" => none_match = Some(value.to_owned()),
                _ => {}
            }
        }
        let s = served.lock().unwrap();
        let head = |status: &str, len: usize, extra: &str| {
            format!("HTTP/1.1 {status}\r\nContent-Length: {len}\r\n{extra}\r\n")
        };
        let etag = format!("ETag: {}\r\n", s.etag);
        if path == "/m/" && method == "GET" {
            let listing = format!(
                r#"[{{ "name":"{REFRESH_FILE}", "type":"file", "mtime":"{}", "size":{} }}]"#,
                s.mtime,
                body.len()
            );
            let _ = write!(stream, "{}{listing}", head("200 OK", listing.len(), ""));
        } else if path == format!("/m/{REFRESH_FILE}") {
            let status = if method == "HEAD" {
                if s.honour_none_match && none_match.as_deref() == Some(s.etag.as_str()) {
                    let _ = write!(stream, "{}", head("304 Not Modified", 0, &etag));
                    304
                } else {
                    let _ = write!(stream, "{}", head("200 OK", body.len(), &etag));
                    200
                }
            } else if let Some((a, b)) = range {
                let extra = format!("{etag}Content-Range: bytes {a}-{b}/{}\r\n", body.len());
                let _ = write!(stream, "{}", head("206 Partial Content", b + 1 - a, &extra));
                let _ = stream.write_all(&body[a..=b]);
                206
            } else {
                let _ = write!(stream, "{}", head("200 OK", body.len(), &etag));
                let _ = stream.write_all(body);
                200
            };
            seen.lock().unwrap().push(Seen {
                method,
                none_match,
                status,
            });
        } else {
            let _ = write!(stream, "{}", head("404 Not Found", 0, ""));
        }
        drop(s);
        let _ = stream.flush();
    }
}

/// Adds the refresh server's nginx JSON listing as source 1 and waits for the probe's
/// information, read with the ETag kept beside it (Q-230).
fn nginx_source(t: &mut TestApp, server: &RefreshServer) {
    add_and_wait(
        t,
        SourceSpec {
            kind: "http".into(),
            location: server.base.clone(),
            name: "refresh".into(),
            flavor: Some(ListingFlavor::NginxJson),
        },
    );
}

/// Refreshes source 1 and advances frames until `done` holds of the probe's requests from
/// record `from` on, at most 2000 frames.
fn refresh_until(
    t: &mut TestApp,
    server: &RefreshServer,
    from: usize,
    done: impl Fn(&[Seen]) -> bool,
) -> bool {
    t.command(LibraryCommand::Refresh(1));
    pump_until(t, 2000, |_| done(&server.seen()[from..]))
}

/// A header read: a `GET` of the probe.
fn header_read(seen: &[Seen]) -> bool {
    seen.iter().any(|s| s.method == "GET")
}

/// The refresh's ETag check: a `HEAD` sent with `If-None-Match` (Q-230).
fn etag_check(seen: &[Seen]) -> Option<&Seen> {
    seen.iter()
        .find(|s| s.method == "HEAD" && s.none_match.is_some())
}

// Q-231
#[test]
fn a_changed_nginx_mtime_drops_the_information_and_reads_it_again() {
    let server = RefreshServer::start();
    let mut t = TestApp::new("refresh-mtime");
    nginx_source(&mut t, &server);
    // The same listing: the entry keeps its information and is checked, not read.
    let before = server.seen().len();
    assert!(refresh_until(&mut t, &server, before, |s| etag_check(s).is_some()));
    t.until(200, |_| false);
    assert!(probe_has_info(t.app.world().resource::<Library>()));
    assert!(
        !header_read(&server.seen()[before..]),
        "{:?}",
        server.seen()
    );
    // A new mtime, the size, the ETag and the bytes unchanged: the information is dropped
    // and the header read again, with no ETag check of the old information.
    server.change(|s| s.mtime = "Fri, 30 Oct 2026 12:00:00 GMT".into());
    let before = server.seen().len();
    assert!(
        refresh_until(&mut t, &server, before, header_read),
        "the header is read again: {:?}",
        server.seen()
    );
    let seen = server.seen()[before..].to_vec();
    assert!(etag_check(&seen).is_none(), "{seen:?}");
    assert!(
        t.until(2000, |app| probe_has_info(
            app.world().resource::<Library>()
        )),
        "the new read's information arrives"
    );
}

// Q-230
#[test]
fn a_200_with_another_etag_drops_the_information_and_reads_it_again() {
    let server = RefreshServer::start();
    let mut t = TestApp::new("refresh-other-etag");
    nginx_source(&mut t, &server);
    server.change(|s| s.etag = "\"v2\"".into());
    let before = server.seen().len();
    assert!(
        refresh_until(&mut t, &server, before, header_read),
        "the header is read again: {:?}",
        server.seen()
    );
    let seen = server.seen()[before..].to_vec();
    let head = etag_check(&seen).expect("the refresh checks the ETag");
    assert_eq!(head.none_match.as_deref(), Some("\"v1\""));
    assert_eq!(head.status, 200);
    assert!(
        t.until(2000, |app| probe_has_info(
            app.world().resource::<Library>()
        )),
        "the new read's information arrives"
    );
}

// Q-230
#[test]
fn a_200_with_the_same_etag_keeps_the_information() {
    let server = RefreshServer::start();
    let mut t = TestApp::new("refresh-same-etag-200");
    nginx_source(&mut t, &server);
    // A server that ignores `If-None-Match`: 200 with the ETag the entry was kept with.
    server.change(|s| s.honour_none_match = false);
    let before = server.seen().len();
    assert!(refresh_until(&mut t, &server, before, |s| etag_check(s).is_some()));
    let head = etag_check(&server.seen()[before..]).cloned().unwrap();
    assert_eq!(head.status, 200, "{head:?}");
    assert_eq!(head.none_match.as_deref(), Some("\"v1\""), "the ETag kept");
    t.until(200, |_| false);
    assert!(probe_has_info(t.app.world().resource::<Library>()));
    assert!(
        !header_read(&server.seen()[before..]),
        "no header read: {:?}",
        server.seen()
    );
}
