// SPDX-License-Identifier: Apache-2.0

//! I-2 (§1.5 HTTP index, D-31, `cpu`): an HTTP source over the fixture server lists the
//! same entries in the three listing flavors, and an `ftgs-if` entry's header-only
//! information is filled while the bytes the server sent for its file stay below the
//! file's size. A listed directory lists on demand (Q-192).

use crate::support;

use ftgsxr_app::{EntryState, Library, LibraryCommand};
use ftgsxr_io::{EntryKind, ListingFlavor, SourceSpec};
use ftgsxr_testkit::ServedListing;
use support::TestApp;

/// The probe's path on the server, as sent, and its size.
const PROBE: &str = "/listing/models/probe%20p1.safetensors";
const PROBE_SIZE: u64 = 1092;

/// Adds the fixture server's listing base as source 1 in `flavor` and waits for its
/// entries and the probe's information.
fn listed(t: &mut TestApp, flavor: ListingFlavor) -> Vec<(String, String)> {
    let base = t.server.url_of("listing/models/");
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: base,
        name: "fixtures".into(),
        flavor: Some(flavor),
    }));
    let ok = t.until(2000, |app| {
        let lib = app.world().resource::<Library>();
        lib.entries
            .iter()
            .any(|e| e.source == 1 && e.entry.kind == EntryKind::Exchange && e.info.is_some())
    });
    assert!(
        ok,
        "{flavor:?}: the source lists and the probe's information arrives"
    );
    // Each app has its own server: the locations are compared below its root.
    let root = t.server.url();
    let lib = t.app.world().resource::<Library>();
    let mut entries: Vec<(String, String)> = lib
        .entries
        .iter()
        .filter(|e| e.source == 1)
        .map(|e| {
            assert_eq!(e.state, EntryState::Listed);
            (
                e.entry.name.clone(),
                e.entry.location.trim_start_matches(&root).to_owned(),
            )
        })
        .collect();
    entries.sort();
    entries
}

fn probe_bytes(t: &TestApp) -> u64 {
    t.server
        .requests()
        .iter()
        .filter(|r| r.path == PROBE)
        .map(|r| r.body_bytes)
        .sum()
}

#[test]
fn i02_three_flavors_list_the_same_entries_with_header_only_information() {
    let flavors = [
        (ServedListing::Index, ListingFlavor::Index),
        (ServedListing::NginxJson, ListingFlavor::NginxJson),
        (ServedListing::Html, ListingFlavor::Html),
    ];
    let mut lists = Vec::new();
    for (served, flavor) in flavors {
        let mut t = TestApp::new(&format!("i02-{flavor:?}"));
        t.server.serve_listing(served);
        let entries = listed(&mut t, flavor);
        let sent = probe_bytes(&t);
        assert!(
            sent > 0 && sent < PROBE_SIZE,
            "{flavor:?}: {sent} bytes of the probe sent"
        );
        let lib = t.app.world().resource::<Library>();
        let probe = lib
            .entries
            .iter()
            .find(|e| e.entry.kind == EntryKind::Exchange)
            .unwrap();
        let info = probe.info.as_ref().unwrap();
        assert_eq!(info.n, 1, "{flavor:?}: the probe's N");
        lists.push(entries);
    }
    assert_eq!(lists[0].len(), 3, "{:?}", lists[0]);
    assert!(lists[0].iter().any(|(n, _)| n == "tiny"));
    assert_eq!(lists[0], lists[1], "index and nginx JSON");
    assert_eq!(lists[0], lists[2], "index and HTML");
}

// Q-192
#[test]
fn i02_a_listed_directory_lists_on_demand() {
    let mut t = TestApp::new("i02-subdir");
    t.server.serve_listing(ServedListing::Html);
    listed(&mut t, ListingFlavor::Html);
    let dir = {
        let lib = t.app.world().resource::<Library>();
        lib.entries
            .iter()
            .position(|e| e.entry.kind == EntryKind::Dir)
            .expect("the listing's directory")
    };
    let before = t.server.requests().len();
    t.command(LibraryCommand::Load {
        entry: dir,
        reduce: false,
    });
    let ok = t.until(2000, |app| {
        let lib = app.world().resource::<Library>();
        matches!(lib.entries[dir].state, EntryState::Failed(_))
    });
    // The fixture server lists only the base: the directory's own listing fails, and
    // the directory's entry carries the failure; what matters is that it was requested.
    assert!(ok, "the directory's listing ends");
    let asked = t.server.requests()[before..]
        .iter()
        .any(|r| r.path.starts_with("/listing/models/tiny/"));
    assert!(asked, "the directory's listing was requested");
}
