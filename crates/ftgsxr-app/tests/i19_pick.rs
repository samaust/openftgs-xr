// SPDX-License-Identifier: Apache-2.0

//! I-19's picker clause (§1.5 Picked files, §2.5, `cpu`): a stub `Platform` whose
//! `poll_pick` returns two files loads both from `Pick(Files)`; a picked folder joins as
//! a source. Without a GPU "loads" means `LoadedModel` (Q-148).

use crate::support;

use std::fs;

use ftgsxr_app::{Library, LibraryCommand, LoadedModel, ModelRef, PickKind};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

#[test]
fn i19_two_picked_files_load() {
    let mut t = TestApp::new("i19-pick");
    let files: Vec<ModelRef> = ["P1", "P3"]
        .iter()
        .map(|p| ModelRef::File(fixtures_dir().join(format!("probes/{p}.safetensors"))))
        .collect();
    t.platform.queue_pick(Ok(files.clone()));
    t.command(LibraryCommand::Pick(PickKind::Files));
    let ok = t.until(600, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 2
    });
    assert!(ok, "both load");
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    let mut got: Vec<ModelRef> = q.iter(t.app.world()).map(|m| m.source.clone()).collect();
    got.sort_by_key(|r| format!("{r:?}"));
    assert_eq!(got, files);
    assert_eq!(*t.platform.opened.lock().unwrap(), [PickKind::Files]);
}

#[test]
fn i19_a_picked_folder_joins_as_a_source() {
    let mut t = TestApp::new("i19-folder");
    let folder = t.dir.join("picked");
    fs::create_dir_all(&folder).unwrap();
    fs::copy(
        fixtures_dir().join("probes/P1.safetensors"),
        folder.join("P1.safetensors"),
    )
    .unwrap();
    t.platform
        .queue_pick(Ok(vec![ModelRef::File(folder.clone())]));
    t.command(LibraryCommand::Pick(PickKind::Folder));
    let ok = t.until(600, |app| {
        let lib = app.world().resource::<Library>();
        lib.sources.len() == 2 && lib.entries.iter().any(|e| e.source == 1)
    });
    assert!(ok, "the folder is listed");
    let lib = t.app.world().resource::<Library>();
    assert_eq!(lib.sources[1].0.location, folder.to_str().unwrap());
    assert_eq!(
        lib.entries
            .iter()
            .find(|e| e.source == 1)
            .unwrap()
            .entry
            .name,
        "P1"
    );
}
