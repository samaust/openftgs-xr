// SPDX-License-Identifier: Apache-2.0

//! I-21 (D-20, `net` lane): the HTTPS listing of a public index, served under a
//! certificate from a public authority, validates with the WebPki roots and lists. The
//! index is the owner's, named by `FTGSXR_NET_INDEX` (Q-156).

use crate::support;

use ftgsxr_app::{Library, LibraryCommand};
use ftgsxr_io::{ListingFlavor, SourceSpec};
use support::TestApp;

// Q-156
#[test]
#[ignore = "net"]
fn net_i21_the_public_https_index_lists_with_the_webpki_roots() {
    let Ok(base) = std::env::var("FTGSXR_NET_INDEX") else {
        panic!("FTGSXR_NET_INDEX is unset: set it to the public index's base URL (Q-156)");
    };
    assert!(base.starts_with("https://"), "FTGSXR_NET_INDEX: {base}");
    let mut t = TestApp::new("i21-net");
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "http".into(),
        location: base,
        name: "public".into(),
        flavor: Some(ListingFlavor::Index),
    }));
    let ok = t.until(30_000, |app| {
        app.world()
            .resource::<Library>()
            .entries
            .iter()
            .any(|e| e.source == 1)
    });
    assert!(ok, "the public index lists");
}
