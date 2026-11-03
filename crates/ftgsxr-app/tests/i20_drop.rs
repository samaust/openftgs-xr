// SPDX-License-Identifier: Apache-2.0

//! I-20 (§1.7 dropped files, `cpu`): a dropped-file window message loads the file; a
//! dropped non-model reports E-FILE-FORMAT, through `SplatEvent::Failed` since a drop
//! has no library entry (Q-T278.2).

mod support;

use std::fs;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::window::FileDragAndDrop;
use ftgsxr_app::{LoadedModel, ModelRef, SplatEvent};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

fn drop_file(t: &mut TestApp, path: std::path::PathBuf) {
    t.app
        .world_mut()
        .write_message(FileDragAndDrop::DroppedFile {
            window: Entity::PLACEHOLDER,
            path_buf: path,
        });
}

#[test]
fn i20_a_dropped_model_loads() {
    let mut t = TestApp::new("i20-model");
    let path = fixtures_dir().join("probes/P2.safetensors");
    drop_file(&mut t, path.clone());
    let ok = t.until(600, |app| {
        app.world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1
    });
    assert!(ok, "the dropped file loads");
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    let m = q.single(t.app.world()).unwrap();
    assert_eq!(m.source, ModelRef::File(path));
    assert_eq!(m.name, "P2");
}

#[test]
fn i20_a_dropped_non_model_reports_e_file_format() {
    let mut t = TestApp::new("i20-other");
    let errors = Arc::new(Mutex::new(Vec::new()));
    let seen = errors.clone();
    t.app.add_observer(move |e: On<SplatEvent>| {
        if let SplatEvent::Failed { error, .. } = e.event() {
            seen.lock().unwrap().push(error.clone());
        }
    });
    let path = t.dir.join("notes.txt");
    fs::write(&path, "not a model\n").unwrap();
    drop_file(&mut t, path);
    assert!(
        t.until(600, |_| !errors.lock().unwrap().is_empty()),
        "the load fails"
    );
    let text = errors.lock().unwrap()[0].clone();
    assert!(text.starts_with("E-FILE-FORMAT: "), "{text}");
    t.frames(2);
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    assert_eq!(q.iter(t.app.world()).count(), 0);
}
