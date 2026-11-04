// SPDX-License-Identifier: Apache-2.0

//! I-1 (§1.5 Loading and unloading, `gpu` lane): two probes loaded by `LoadRef` become
//! instances with `LoadedModel`, each raising `SplatEvent::Uploaded`; `Unload` removes
//! them and frees their bytes (PRD #275's reading of the row; Q-148).

use crate::support;

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use ftgsxr_app::{LibraryCommand, LoadedModel, ModelRef, SplatEvent};
use ftgsxr_render::SplatGpu;
use ftgsxr_testkit::{fixtures_dir, layer_messages};
use support::TestApp;

#[test]
#[ignore = "gpu"]
fn gpu_i01_two_probes_load_and_unload() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("i01");
    let uploaded = Arc::new(Mutex::new(Vec::new()));
    let seen = uploaded.clone();
    t.app.add_observer(move |e: On<SplatEvent>| {
        if let SplatEvent::Uploaded { entity, .. } = e.event() {
            seen.lock().unwrap().push(*entity);
        }
    });
    for p in ["P1", "P2"] {
        let path = fixtures_dir().join(format!("probes/{p}.safetensors"));
        t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    }
    assert!(
        t.until(600, |_| uploaded.lock().unwrap().len() == 2),
        "two uploads"
    );
    let mut q = t
        .app
        .world_mut()
        .query::<(Entity, &LoadedModel, &Visibility)>();
    let instances: Vec<(Entity, String, Visibility)> = q
        .iter(t.app.world())
        .map(|(e, m, v)| (e, m.name.clone(), *v))
        .collect();
    assert_eq!(instances.len(), 2);
    let mut names: Vec<_> = instances.iter().map(|i| i.1.clone()).collect();
    names.sort();
    assert_eq!(names, ["P1", "P2"]);
    for (e, _, v) in &instances {
        assert!(uploaded.lock().unwrap().contains(e));
        assert_eq!(*v, Visibility::Visible);
    }
    let in_use = |app: &App| {
        app.world()
            .resource::<SplatGpu>()
            .device()
            .model_bytes_in_use()
    };
    assert!(in_use(&t.app) > 0);
    for (e, _, _) in &instances {
        t.command(LibraryCommand::Unload(*e));
    }
    assert!(t.until(60, |app| in_use(app) == 0), "the unload frees");
    let mut q = t.app.world_mut().query::<&LoadedModel>();
    assert_eq!(q.iter(t.app.world()).count(), 0);
}
