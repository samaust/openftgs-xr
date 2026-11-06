// SPDX-License-Identifier: Apache-2.0

//! The environment mode's headset gate (`cpu`, #567; source of truth §1.9 Environment
//! modes, §1.4 Backgrounds "the desktop is always in VR mode", D-17, Q-335, Q-228):
//! `[environment] mode` counts as passthrough only under the `Headset` marker. The floor
//! grid spawns without a window wherever the mesh and material stores exist; under the
//! marker it is hidden in passthrough mode and shown in VR mode with `[display] grid` on,
//! and stick locomotion is off in passthrough mode; without the marker both stay as in VR
//! mode whatever the setting says.

use crate::support;

use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use ftgsxr_app::{Headset, Locomotion, SettingsRes};
use support::{Setup, TestApp};

/// The CPU app with the mesh and material stores, without a window, with the marker
/// when `headset`.
#[track_caller]
fn app(name: &str, headset: bool) -> TestApp {
    TestApp::with(name, Setup::default(), |app| {
        app.init_asset::<Mesh>().init_asset::<StandardMaterial>();
        if headset {
            app.insert_resource(Headset);
        }
    })
}

/// The visibility of every line-list mesh (the floor grid, §1.4).
fn grids(t: &mut TestApp) -> Vec<Visibility> {
    let w = t.app.world_mut();
    let mut q = w.query::<(&Mesh3d, &Visibility)>();
    let rows: Vec<_> = q.iter(w).map(|(m, v)| (m.0.clone(), *v)).collect();
    let meshes = w.resource::<Assets<Mesh>>();
    rows.into_iter()
        .filter(|(m, _)| {
            meshes
                .get(m)
                .is_some_and(|m| m.primitive_topology() == PrimitiveTopology::LineList)
        })
        .map(|(_, v)| v)
        .collect()
}

fn set_mode(t: &mut TestApp, mode: &str, grid: bool) {
    {
        let mut s = t.app.world_mut().resource_mut::<SettingsRes>();
        s.0.environment.mode = mode.into();
        s.0.display.grid = grid;
    }
    t.frames(2);
}

fn shown(t: &mut TestApp) -> bool {
    let g = grids(t);
    assert_eq!(g.len(), 1, "one floor grid: {g:?}");
    g[0] != Visibility::Hidden
}

fn sticks(t: &TestApp) -> bool {
    t.app.world().resource::<Locomotion>().enabled
}

/// Q-228's detecting test: the grid exists without a window, and under the marker it
/// follows the gated mode and `[display] grid`.
#[test]
fn the_grid_spawns_without_a_window_and_hides_in_passthrough_under_the_marker() {
    let mut t = app("grid_headset", true);
    t.frames(1);
    assert!(shown(&mut t), "VR mode with the grid on: shown");
    assert!(sticks(&t));
    set_mode(&mut t, "passthrough", true);
    assert!(!shown(&mut t), "passthrough mode: no background, no grid");
    assert!(!sticks(&t), "D-17: no stick locomotion in passthrough mode");
    set_mode(&mut t, "vr", true);
    assert!(shown(&mut t), "back in VR mode: shown");
    assert!(sticks(&t));
    set_mode(&mut t, "vr", false);
    assert!(!shown(&mut t), "`[display] grid` off: hidden");
}

/// Without the marker (the desktop, `play --xr`) the player stays in VR mode whatever
/// `[environment] mode` says (Q-335).
#[test]
fn without_the_marker_passthrough_in_the_settings_keeps_vr_mode() {
    let mut t = app("grid_desktop", false);
    t.frames(1);
    assert!(shown(&mut t));
    set_mode(&mut t, "passthrough", true);
    assert!(shown(&mut t), "the desktop is always in VR mode");
    assert!(sticks(&t), "the sticks stay on");
    set_mode(&mut t, "passthrough", false);
    assert!(!shown(&mut t), "`[display] grid` still applies");
}

/// A settings file asking for passthrough at startup starts the headset in it (the
/// mapping before `RigPlugin` builds, Q-195).
#[test]
fn a_saved_passthrough_mode_starts_the_headset_without_the_grid() {
    let mut t = TestApp::with("grid_startup", Setup::default(), |app| {
        app.init_asset::<Mesh>().init_asset::<StandardMaterial>();
        app.insert_resource(Headset);
        let mut s = ftgsxr_app::Settings::default();
        s.environment.mode = "passthrough".into();
        app.insert_resource(SettingsRes(s));
    });
    t.frames(1);
    assert!(!shown(&mut t));
    assert!(!sticks(&t));
}

/// `Toggle(Feature::Occlusion)` flips `[environment] occlusion` in the settings, as the
/// other toggles do; a second one flips it back (Q-336, Q-207).
#[test]
fn toggle_occlusion_flips_the_setting() {
    use ftgsxr_app::{Feature, Intent};
    let mut t = app("toggle_occlusion", true);
    t.frames(1);
    let occlusion = |t: &TestApp| {
        t.app
            .world()
            .resource::<SettingsRes>()
            .0
            .environment
            .occlusion
    };
    let before = occlusion(&t);
    t.app
        .world_mut()
        .write_message(Intent::Toggle(Feature::Occlusion));
    t.frames(2);
    assert_eq!(occlusion(&t), !before);
    t.app
        .world_mut()
        .write_message(Intent::Toggle(Feature::Occlusion));
    t.frames(2);
    assert_eq!(occlusion(&t), before);
}
