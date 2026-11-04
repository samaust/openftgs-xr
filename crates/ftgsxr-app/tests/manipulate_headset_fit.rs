// SPDX-License-Identifier: Apache-2.0

//! D-11's automatic fit on the headset (`cpu`, #455; source of truth §1.5 Poses: "on the
//! headset the fit is applied automatically on the first load of a model without a saved
//! pose when B_m's longest side lies outside [0.1 m, 5 m]"; "a saved pose wins over the
//! identity and the auto-fit"; D-11; I-4's fit and tolerance; Q-266: the headset is the
//! `Headset` marker; Q-180: the head without a camera is the stage origin facing −z;
//! Q-T455.3).

use crate::support::{Setup, TestApp};

use bevy::prelude::*;
use ftgsxr_app::{ContentGroup, Headset, LibraryCommand, LoadedModel, ModelRef, Poses};
use ftgsxr_core::{Aabb, Pose, Preset};
use ftgsxr_io::{PoseStore, SavedPose};
use ftgsxr_render::{SplatAsset, SplatModel};

/// I-4's tolerance.
const TOL: f32 = 1e-4;

/// An exchange model whose rows lie along x over `len_m`, small and still, so B_m's
/// longest side is `len_m` give or take a few millimetres, written into its own folder
/// (no manifest beside it: Y-up, L_fit = 1 m).
fn model_file(t: &TestApp, len_m: f32) -> std::path::PathBuf {
    let mut m = ftgsxr_testkit::random_model(11, 8, 0);
    let n = m.position.len();
    for i in 0..n {
        m.position[i] = [-0.5 * len_m + len_m * i as f32 / (n - 1) as f32, 0.0, 0.0];
        m.velocity[i] = [0.0; 3];
        m.scale[i] = [0.0005; 3];
        m.opacity[i] = 0.9;
    }
    let dir = t.dir.join(format!("model_{len_m}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&m)).unwrap();
    path
}

fn app(name: &str, headset: bool, poses: Option<PoseStore>) -> TestApp {
    TestApp::with(name, Setup::default(), |app| {
        if headset {
            app.insert_resource(Headset);
        }
        if let Some(store) = poses {
            app.insert_resource(Poses(store));
        }
    })
}

fn load(t: &mut TestApp, path: std::path::PathBuf) -> Entity {
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "the model loads"
    );
    t.frames(2);
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    q.single(t.app.world()).unwrap()
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

fn bounds(t: &TestApp, e: Entity) -> Aabb {
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    t.app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
        .bounds
}

fn is_identity(p: Pose) -> bool {
    p.translation.length() < TOL
        && (p.rotation * Vec3::X - Vec3::X).length() < TOL
        && (p.rotation * Vec3::Y - Vec3::Y).length() < TOL
        && (p.scale - 1.0).abs() < TOL
}

/// I-4's fit without a camera: the longest posed side at L_fit = 1 m, the bottom on
/// y = 0, the center 1.5 m along −z.
fn assert_fitted(b: &Aabb, p: Pose) {
    assert!(
        (b.size().max_element() * p.scale - 1.0).abs() < TOL,
        "{p:?}"
    );
    let bottom = (0..8)
        .map(|k| {
            p.transform_point(Vec3::new(
                if k & 1 == 0 { b.min.x } else { b.max.x },
                if k & 2 == 0 { b.min.y } else { b.max.y },
                if k & 4 == 0 { b.min.z } else { b.max.z },
            ))
            .y
        })
        .fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < TOL, "{bottom}");
    let c = p.transform_point(b.center());
    assert!(c.x.abs() < TOL && (c.z + 1.5).abs() < TOL, "{c}");
}

#[test]
fn headset_fit_a_20_m_model_is_fitted_on_its_first_load() {
    // Under a yaw-only content group moved off the origin (Q-254): the fit is a stage
    // pose, written relative to the group (Q-247).
    let mut t = app("headset_fit_20", true, None);
    {
        let mut q = t
            .app
            .world_mut()
            .query_filtered::<&mut Transform, With<ContentGroup>>();
        *q.single_mut(t.app.world_mut()).unwrap() = Transform {
            translation: Vec3::new(2.0, 0.4, -1.0),
            rotation: Quat::from_rotation_y(50f32.to_radians()),
            scale: Vec3::ONE,
        };
    }
    let path = model_file(&t, 20.0);
    let e = load(&mut t, path);
    let b = bounds(&t, e);
    assert!(b.size().max_element() > 5.0, "{b:?}");
    let (s, rotation, translation) = t
        .app
        .world()
        .get::<GlobalTransform>(e)
        .unwrap()
        .to_scale_rotation_translation();
    assert_fitted(
        &b,
        Pose {
            translation,
            rotation,
            scale: s.x,
        },
    );
}

#[test]
fn headset_fit_a_1_m_model_is_not_fitted() {
    let mut t = app("headset_fit_1", true, None);
    let path = model_file(&t, 1.0);
    let e = load(&mut t, path);
    let b = bounds(&t, e);
    assert!((0.1..=5.0).contains(&b.size().max_element()), "{b:?}");
    assert!(is_identity(pose(&t, e)), "{:?}", pose(&t, e));
}

#[test]
fn headset_fit_a_saved_pose_wins() {
    let id = {
        let mut t = app("headset_fit_saved_id", false, None);
        let path = model_file(&t, 20.0);
        let e = load(&mut t, path);
        t.app
            .world()
            .get::<LoadedModel>(e)
            .unwrap()
            .identity
            .clone()
    };
    let saved = Pose {
        translation: Vec3::new(1.0, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: 0.5,
    };
    let mut store = PoseStore::default();
    store.set(
        &id,
        SavedPose {
            pose: saved,
            preset: Preset::YUp,
            fit_size_m: 1.0,
            saved_at: "2026-11-04T00:00:00.000000000Z".into(),
        },
    );
    let mut t = app("headset_fit_saved", true, Some(store));
    let path = model_file(&t, 20.0);
    let e = load(&mut t, path);
    let p = pose(&t, e);
    assert!(
        (p.translation - saved.translation).length() < TOL && (p.scale - 0.5).abs() < TOL,
        "{p:?}"
    );
}

#[test]
fn headset_fit_without_the_marker_nothing_is_fitted() {
    for len in [20.0, 1.0] {
        let mut t = app(&format!("headset_fit_none_{len}"), false, None);
        let path = model_file(&t, len);
        let e = load(&mut t, path);
        assert!(is_identity(pose(&t, e)), "{len}: {:?}", pose(&t, e));
    }
}
