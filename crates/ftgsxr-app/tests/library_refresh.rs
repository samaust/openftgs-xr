// SPDX-License-Identifier: Apache-2.0

//! Added for #375 (`cpu`): a library refresh keeps an entry's information only while its
//! file is unchanged, the ETag of an HTTP entry included (§1.5 Library: "entries whose file
//! changed (size, ETag or SHA-256) lose their cached information"; Q-229, Q-231). No §3 row
//! pins a refresh; §1.5's sentence is the source of truth.

mod support;

use ftgsxr_app::{Library, LibraryCommand};
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

// Q-231
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
