// SPDX-License-Identifier: Apache-2.0

//! I-23, trim in the app (§3.3; §1.5, §1.7, E17, D-56), `cpu` lane: `SetTrim` with a box
//! gives `SplatBounds` the trimmed boxes once computed, the instance does not move, and
//! the next fit (B_c under `fit_to` content, B_m under whole) and the ray hit use them;
//! an empty trim gives the shape's box with `kept_none`; `ResetTrim` turns it off with the
//! initial shapes; Save pose then reload restores pose and trim; Reset pose leaves the
//! trim; Forget pose drops the saved trim; scene save and load restore it; the asset's
//! box, the thumbnail's, stays untrimmed. The panel's labels and notice are the unit tests
//! `ui::dock::tests::i23_*` (they need the crate's own dock).

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    Hand, Hovered, InstanceAction, Intent, LibraryCommand, LoadedModel, ManipulateSettings,
    ModelRef, PersistCommand, Poses, scene_from_world,
};
use ftgsxr_core::{
    Aabb, Model, ModelKind, Pose, RenderConventions, TimeSpan, Trim, TrimMode, Vec3 as CVec3,
};
use ftgsxr_render::{SplatAsset, SplatBounds, SplatModel, SplatTrim};
use support::{StubPlatform, TestApp};

/// Three rows near the origin and a floater at x = 10, all still, scale 0.01.
fn floater_model() -> Model {
    let at = [
        [-0.5, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.5, 0.0, 0.1],
        [10.0, 0.0, 0.0],
    ];
    let n = at.len();
    Model {
        position: at.to_vec(),
        time_center: vec![0.5; n],
        duration: vec![1000.0; n],
        velocity: vec![[0.0; 3]; n],
        scale: vec![[0.01; 3]; n],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacity: vec![0.5; n],
        sh: vec![0.0; n * 3],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 1.0,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: "i23".into(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// The box that keeps the three rows near the origin.
fn near_box() -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: CVec3::splat(-1.0),
        box_max: CVec3::splat(1.0),
        center: CVec3::ZERO,
        radius: 1.0,
    }
}

/// E17's trimmed boxes of `floater_model` under `near_box`, by hand: B_m the three rows
/// ± 0.03, B_c their own box (three points: the 1st and 99th percentiles are the extremes).
fn trimmed_by_hand() -> (Aabb, Aabb) {
    (
        Aabb {
            min: CVec3::new(-0.53, -0.03, -0.03),
            max: CVec3::new(0.53, 0.23, 0.13),
        },
        Aabb {
            min: CVec3::new(-0.5, 0.0, 0.0),
            max: CVec3::new(0.5, 0.2, 0.1),
        },
    )
}

fn model_file(name: &str) -> std::path::PathBuf {
    let dir = support::scratch(&format!("{name}_model"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("floater.safetensors");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&floater_model())).unwrap();
    path
}

fn load(t: &mut TestApp, path: std::path::PathBuf) -> Entity {
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&SplatBounds>()
            .iter(app.world())
            .count()
            == 1),
        "the model loads and has its boxes"
    );
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    q.single(t.app.world()).unwrap()
}

fn act(t: &mut TestApp, instance: Entity, action: InstanceAction) {
    t.app
        .world_mut()
        .write_message(Intent::Instance { instance, action });
    t.frames(1);
}

fn bounds(t: &TestApp, e: Entity) -> SplatBounds {
    *t.app.world().get::<SplatBounds>(e).expect("its boxes")
}

fn trim(t: &TestApp, e: Entity) -> Option<Trim> {
    t.app.world().get::<SplatTrim>(e).map(|s| s.0)
}

fn asset(t: &TestApp, e: Entity) -> &SplatAsset {
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    t.app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
}

/// `SetTrim`, then the frames until the boxes become `want`'s B_m.
fn set_trim_and_wait(t: &mut TestApp, e: Entity, tr: Trim, want: Aabb) {
    act(t, e, InstanceAction::SetTrim(tr));
    assert!(
        t.until(600, |app| app
            .world()
            .get::<SplatBounds>(e)
            .is_some_and(|b| close(b.aabb, want))),
        "the trimmed boxes arrive"
    );
}

fn close(a: Aabb, b: Aabb) -> bool {
    (a.min - b.min).abs().max_element() <= 1e-6 && (a.max - b.max).abs().max_element() <= 1e-6
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

fn hovered_by(t: &mut TestApp, e: Entity, origin: Vec3) -> bool {
    t.app.world_mut().write_message(Intent::Pointer {
        hand: Hand::Right,
        origin,
        dir: -Vec3::Z,
    });
    t.frames(1);
    t.app.world().get::<Hovered>(e).is_some()
}

#[test]
fn i23_a_box_trim_gives_the_trimmed_boxes_the_fit_and_the_ray_use() {
    let mut t = TestApp::new("i23_box");
    let e = load(&mut t, model_file("i23_box"));
    let untrimmed = asset(&t, e).bounds;
    assert!(close(bounds(&t, e).aabb, untrimmed), "untrimmed at first");
    assert!(!bounds(&t, e).kept_none);
    // A ray down −z at x = 5 hits the untrimmed box only (the floater widens it).
    let at_floater_gap = Vec3::new(5.0, 0.0, 3.0);
    assert!(hovered_by(&mut t, e, at_floater_gap));
    let before = pose(&t, e);
    let (want_m, want_c) = trimmed_by_hand();
    set_trim_and_wait(&mut t, e, near_box(), want_m);
    let b = bounds(&t, e);
    assert!(close(b.content, want_c), "{:?}", b.content);
    assert!(!b.kept_none);
    // The trim never moves the instance; the asset's box (the thumbnail's) is untrimmed.
    assert_eq!(pose(&t, e), before);
    assert_eq!(asset(&t, e).bounds, untrimmed);
    // The ray at x = 5 now misses; one through the kept rows hits.
    assert!(!hovered_by(&mut t, e, at_floater_gap));
    assert!(hovered_by(&mut t, e, Vec3::new(0.0, 0.1, 3.0)));
    // The fit under `fit_to` content sizes B_c's longest side (1.0) to 1 m.
    act(&mut t, e, InstanceAction::Fit);
    let p = pose(&t, e);
    assert!(
        (p.scale * want_c.longest_side() - 1.0).abs() < 1e-4,
        "{p:?}"
    );
    let c = p.transform_point(want_c.center());
    assert!(c.x.abs() < 1e-4 && (c.z + 1.5).abs() < 1e-4, "{c}");
    // Under whole, B_m's (1.06).
    t.app
        .world_mut()
        .resource_mut::<ManipulateSettings>()
        .fit_to_content = false;
    act(&mut t, e, InstanceAction::Fit);
    let p = pose(&t, e);
    assert!(
        (p.scale * want_m.longest_side() - 1.0).abs() < 1e-4,
        "{p:?}"
    );
}

#[test]
fn i23_an_empty_trim_gives_the_shape_box_and_reset_trim_turns_it_off() {
    let mut t = TestApp::new("i23_empty");
    let e = load(&mut t, model_file("i23_empty"));
    let untrimmed = asset(&t, e).bounds;
    let far = Trim {
        mode: TrimMode::Sphere,
        center: CVec3::new(0.0, 5.0, 0.0),
        radius: 0.5,
        ..near_box()
    };
    let shape = far.shape_bounds().unwrap();
    set_trim_and_wait(&mut t, e, far, shape);
    let b = bounds(&t, e);
    assert!(b.kept_none);
    assert_eq!((b.aabb, b.content), (shape, shape));
    // Reset trim: off, with the initial shapes of the untrimmed B_m, and the untrimmed
    // boxes back.
    act(&mut t, e, InstanceAction::ResetTrim);
    t.frames(1);
    assert_eq!(trim(&t, e), Some(Trim::initial(&untrimmed)));
    assert_eq!(trim(&t, e).unwrap().mode, TrimMode::Off);
    let b = bounds(&t, e);
    assert!(close(b.aabb, untrimmed) && !b.kept_none);
}

#[test]
fn i23_save_pose_keeps_the_trim_reset_and_forget_pose_leave_it() {
    let path = model_file("i23_poses");
    let mut t = TestApp::new("i23_poses");
    let e = load(&mut t, path.clone());
    let id = t
        .app
        .world()
        .get::<LoadedModel>(e)
        .unwrap()
        .identity
        .clone();
    let (want_m, _) = trimmed_by_hand();
    set_trim_and_wait(&mut t, e, near_box(), want_m);
    let moved = Transform::from_xyz(0.3, 0.0, -1.0).with_scale(Vec3::splat(2.0));
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = moved;
    t.app.world_mut().write_message(PersistCommand::SavePose(e));
    t.frames(3);
    let saved = t
        .app
        .world()
        .resource::<Poses>()
        .0
        .get(&id)
        .cloned()
        .unwrap();
    assert_eq!(saved.trim, Some(near_box()));
    // Reset pose leaves the trim alone.
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = Transform::IDENTITY;
    act(&mut t, e, InstanceAction::ResetPose);
    assert_eq!(trim(&t, e), Some(near_box()));
    assert_eq!(pose(&t, e).translation, moved.translation);
    // A new player on the same directories: the pose and the trim load together.
    let dir = t.dir.clone();
    drop(t);
    let mut again = TestApp::on("i23_poses_reload", StubPlatform::new(&dir));
    let e = load(&mut again, path);
    assert_eq!(trim(&again, e), Some(near_box()));
    assert_eq!(pose(&again, e).translation, moved.translation);
    assert!(
        again.until(600, |app| app
            .world()
            .get::<SplatBounds>(e)
            .is_some_and(|b| close(b.aabb, want_m))),
        "the saved trim's boxes"
    );
    // Forget pose drops the saved trim with the pose; the instance keeps its trim.
    act(&mut again, e, InstanceAction::ForgetPose);
    assert!(again.app.world().resource::<Poses>().0.get(&id).is_none());
    assert_eq!(trim(&again, e), Some(near_box()));
}

#[test]
fn i23_a_scene_saves_and_restores_the_trim() {
    let path = model_file("i23_scene");
    let mut t = TestApp::new("i23_scene");
    let e = load(&mut t, path);
    let (want_m, _) = trimmed_by_hand();
    set_trim_and_wait(&mut t, e, near_box(), want_m);
    assert_eq!(
        scene_from_world(t.app.world()).models[0].trim,
        Some(near_box())
    );
    t.app
        .world_mut()
        .write_message(PersistCommand::SaveScene("trimmed".into()));
    t.frames(1);
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("trimmed".into()));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&SplatTrim>()
            .iter(app.world())
            .any(|s| s.0 == near_box())),
        "the scene's trim is restored"
    );
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    let e = q.single(t.app.world()).unwrap();
    assert!(
        t.until(600, |app| app
            .world()
            .get::<SplatBounds>(e)
            .is_some_and(|b| close(b.aabb, want_m))),
        "and its boxes"
    );
}
