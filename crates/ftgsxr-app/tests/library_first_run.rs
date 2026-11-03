// SPDX-License-Identifier: Apache-2.0

//! A first run (#286, Q-216, Q-14): a missing default models directory lists as an
//! empty source, with no toast and no directory created; any other missing directory is
//! E-IO.

mod support;

use ftgsxr_app::{Library, LibraryCommand};
use ftgsxr_io::SourceSpec;
use support::TestApp;
use support::ui::{record_toasts, toasted, toasts};

// Q-216
#[test]
fn a_missing_default_models_directory_is_an_empty_source() {
    let mut t = TestApp::new("first-run-default");
    record_toasts(&mut t.app);
    let models = t.dir.join("models");
    assert!(!models.exists());
    t.command(LibraryCommand::Refresh(0));
    t.frames(50);
    assert_eq!(toasts(&t.app), vec![]);
    assert!(t.app.world().resource::<Library>().entries.is_empty());
    assert!(!models.exists(), "no directory is created (Q-14)");
}

// Q-216
#[test]
fn another_missing_directory_is_e_io() {
    let mut t = TestApp::new("first-run-other");
    record_toasts(&mut t.app);
    let other = t.dir.join("elsewhere");
    t.command(LibraryCommand::AddSource(SourceSpec {
        kind: "dir".into(),
        location: other.to_string_lossy().into_owned(),
        name: "models".into(),
        flavor: None,
    }));
    assert!(
        t.until(300, |app| toasted(app, "E-IO: ")),
        "{:?}",
        toasts(&t.app)
    );
    assert!(!other.exists());
}
