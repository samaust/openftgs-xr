// SPDX-License-Identifier: Apache-2.0

//! Added for #375 (`cpu`): a library refresh keeps an entry's information only while its
//! file is unchanged, the ETag of an HTTP entry included (§1.5 Library: "entries whose file
//! changed (size, ETag or SHA-256) lose their cached information"; Q-230, Q-232). No §3 row
//! pins a refresh; §1.5's sentence is the source of truth. #400 adds the ETag of the `HEAD`
//! sent before a load (Q-242).

mod support;

use bevy::asset::Handle;
use ftgsxr_app::{Library, LibraryCommand, LibraryEntry, LoadedModel};
use ftgsxr_io::{EntryKind, ListingFlavor, SourceSpec};
use ftgsxr_testkit::{RequestRecord, ServedListing, fixtures_dir};
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
    t.server.requests()[from..]
        .iter()
        .filter(|r| r.path == PROBE)
        .cloned()
        .collect()
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
    let url = ftgsxr_io::Url::parse(&t.server.url_of("listing/models/snapshot.ply")).unwrap();
    t.command(LibraryCommand::LoadRef(ftgsxr_io::ModelRef::Http(url)));
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
