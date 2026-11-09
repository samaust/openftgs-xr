// SPDX-License-Identifier: Apache-2.0

//! I-24, trim handles in the app (§3.3; E18, §1.7, §1.8, D-57), `cpu` lane. Desktop: in
//! `GizmoMode::Trim` a mouse drag of a face handle sends `SetTrim` with the expected
//! limit (read back as the `SplatTrim` it gives), a handle hit wins over the instance
//! pick, even behind a nearer instance hit, there are no handles with the trim
//! off or in another gizmo mode, C is ignored with the trim off and turning the trim off
//! returns the gizmo to translate. XR, from `Pointer`, `Select` and `Grab*` intents: a
//! `grab` drag of a face handle moves it by p = o + t_h·d, a `select` drag of the center
//! moves c, a radius drag stops at the floor, one hand drags one handle, the second hand
//! cannot take the dragged handle but can drag another, the instance cannot be grabbed
//! during a drag and its handles are hidden while it is grabbed, and a deselect or a trim
//! change ends the drag. Added (Q-379, #655): with two hands' drags of one instance,
//! either release keeps the other hand's face, center or radius as it left them, the trim
//! does not jump in the frame a drag starts or ends, and one drag of a face past its
//! opposite and back still returns the opposite to its start value. Which handle and
//! whether a drag runs are read from `TrimHandles` (Q-379) and compared exactly; the
//! trim's values within 10⁻⁶.

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    GizmoMode, Grabbed, Hand, Hovered, InstanceAction, Intent, LibraryCommand, LoadedModel,
    ModelRef, Selected, TrimHandles,
};
use ftgsxr_core::{
    Model, ModelKind, Pose, RenderConventions, TimeSpan, Trim, TrimHandle, TrimMode, Vec3 as CVec3,
};
use ftgsxr_render::{SplatBounds, SplatTrim};
use support::TestApp;

/// I-24's tolerance for the trim's values.
const TOL: f32 = 1e-6;

/// Eight still rows at the corners of [−0.9, 0.9]³, scale 0.01: B_m is [−0.93, 0.93]³,
/// its longest side 1.86, so E18's ρ is 1.86 · 10⁻³.
fn corners_model() -> Model {
    let mut at = Vec::new();
    for x in [-0.9f32, 0.9] {
        for y in [-0.9f32, 0.9] {
            for z in [-0.9f32, 0.9] {
                at.push([x, y, z]);
            }
        }
    }
    let n = at.len();
    Model {
        position: at,
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
        producer: "i24".into(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// E18's ρ for `corners_model`.
const FLOOR: f32 = 1.86e-3;

/// The box [−1, 1]³, which keeps every row.
fn unit_box() -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: CVec3::splat(-1.0),
        box_max: CVec3::splat(1.0),
        center: CVec3::ZERO,
        radius: 1.0,
    }
}

/// The sphere of radius 2 at the origin, which keeps every row (‖(0.9, 0.9, 0.9)‖ < 1.56).
fn wide_sphere() -> Trim {
    Trim {
        mode: TrimMode::Sphere,
        radius: 2.0,
        ..unit_box()
    }
}

const PLUS_X: TrimHandle = TrimHandle::Face { axis: 0, max: true };
const MINUS_X: TrimHandle = TrimHandle::Face {
    axis: 0,
    max: false,
};

/// `corners_model` loaded at the identity pose, selected, its trim `trim`.
fn setup(name: &str, trim: Trim) -> (TestApp, Entity) {
    let mut t = TestApp::new(name);
    let dir = support::scratch(&format!("{name}_model"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corners.safetensors");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&corners_model())).unwrap();
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
    let e = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>()
        .single(t.app.world())
        .unwrap();
    assert_eq!(pose(&t, e), Pose::IDENTITY);
    send(
        &mut t,
        &[
            act(e, InstanceAction::Select),
            act(e, InstanceAction::SetTrim(trim)),
        ],
    );
    assert!(t.app.world().get::<Selected>(e).is_some());
    (t, e)
}

fn act(instance: Entity, action: InstanceAction) -> Intent {
    Intent::Instance { instance, action }
}

/// Writes the intents and runs two frames: a drag's `SetTrim` is written after
/// `manipulate` reads the frame's intents, and applied in the next frame.
fn send(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    t.frames(2);
}

/// Writes the intents and runs one frame: hover is recomputed from each frame's own
/// `Pointer` intents (Q-263).
fn point(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    t.frames(1);
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

fn trim(t: &TestApp, e: Entity) -> Trim {
    t.app.world().get::<SplatTrim>(e).expect("a trim").0
}

fn handles(t: &TestApp, e: Entity) -> TrimHandles {
    t.app
        .world()
        .get::<TrimHandles>(e)
        .cloned()
        .unwrap_or_default()
}

fn dragged(t: &TestApp, e: Entity) -> Vec<(Hand, TrimHandle)> {
    handles(t, e).dragged
}

/// A ray from `origin` down −z.
fn down(hand: Hand, origin: Vec3) -> Intent {
    Intent::Pointer {
        hand,
        origin,
        dir: -Vec3::Z,
    }
}

/// A ray from (3, 3, 3) + `offset` towards the origin: through the instance's box and
/// 0.82 from every face and radius handle on the axes.
fn diagonal(hand: Hand, offset: Vec3) -> Intent {
    Intent::Pointer {
        hand,
        origin: Vec3::splat(3.0) + offset,
        dir: -Vec3::ONE.normalize(),
    }
}

fn press(hand: Hand) -> Intent {
    Intent::Select {
        hand,
        pressed: true,
    }
}

fn release(hand: Hand) -> Intent {
    Intent::Select {
        hand,
        pressed: false,
    }
}

fn grab(hand: Hand) -> Intent {
    Intent::GrabBegin {
        hand,
        pose: Pose::IDENTITY,
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL
}

#[test]
fn i24_desktop_a_face_drag_sets_its_limit_and_the_handle_wins_over_the_pick() {
    let (mut t, e) = setup("i24_desktop_face", unit_box());
    send(&mut t, &[Intent::Gizmo(GizmoMode::Trim)]);
    // Down −z through x = 1: the +x face's handle, outside B_m (x ≤ 0.93), where a pick
    // would hit nothing and deselect.
    send(
        &mut t,
        &[
            down(Hand::Mouse, Vec3::new(1.0, 0.0, 5.0)),
            press(Hand::Mouse),
        ],
    );
    assert_eq!(dragged(&t, e), [(Hand::Mouse, PLUS_X)]);
    assert!(t.app.world().get::<Selected>(e).is_some(), "still selected");
    // The closest point of the face's line (the x axis through (1, 0, 0)) to the ray.
    send(&mut t, &[down(Hand::Mouse, Vec3::new(1.25, 0.3, 5.0))]);
    let tr = trim(&t, e);
    assert!(close(tr.box_max.x, 1.25), "{tr:?}");
    assert_eq!(tr.box_min, CVec3::splat(-1.0));
    assert_eq!((tr.box_max.y, tr.box_max.z), (1.0, 1.0));
    assert_eq!(pose(&t, e), Pose::IDENTITY, "the instance never moves");
    send(&mut t, &[release(Hand::Mouse)]);
    assert!(dragged(&t, e).is_empty());
    send(&mut t, &[down(Hand::Mouse, Vec3::new(2.0, 0.0, 5.0))]);
    assert!(close(trim(&t, e).box_max.x, 1.25), "the drag has ended");
}

/// A ray from (0, 3, 5) to the −z face's handle at (0, 0, −1): it enters B_m at z = 0.93,
/// t ≈ 4.6, before the handle at t ≈ 6.7, and passes 0.45 or more from every other handle.
fn oblique(hand: Hand) -> Intent {
    Intent::Pointer {
        hand,
        origin: Vec3::new(0.0, 3.0, 5.0),
        dir: Vec3::new(0.0, -3.0, -6.0).normalize(),
    }
}

#[test]
fn i24_a_handle_behind_a_nearer_instance_hit_still_wins() {
    let minus_z = TrimHandle::Face {
        axis: 2,
        max: false,
    };
    let (mut t, e) = setup("i24_behind", unit_box());
    send(&mut t, &[Intent::Gizmo(GizmoMode::Trim)]);
    send(&mut t, &[oblique(Hand::Mouse), press(Hand::Mouse)]);
    assert_eq!(dragged(&t, e), [(Hand::Mouse, minus_z)]);
    send(&mut t, &[release(Hand::Mouse)]);
    // A hand's grab takes the handle, not the instance its ray meets first.
    send(&mut t, &[oblique(Hand::Left), grab(Hand::Left)]);
    assert_eq!(dragged(&t, e), [(Hand::Left, minus_z)]);
    assert!(t.app.world().get::<Grabbed>(e).is_none());
}

#[test]
fn i24_desktop_no_handles_in_another_mode_or_with_the_trim_off_and_c_ignored() {
    let (mut t, e) = setup("i24_desktop_modes", unit_box());
    // The scale cube's mode: the handle's ray picks, misses B_m and deselects.
    send(&mut t, &[Intent::Gizmo(GizmoMode::Scale)]);
    point(&mut t, &[down(Hand::Mouse, Vec3::new(1.0, 0.0, 5.0))]);
    assert!(
        handles(&t, e).hovered.is_empty(),
        "no handles in another mode"
    );
    send(&mut t, &[press(Hand::Mouse)]);
    assert!(dragged(&t, e).is_empty());
    assert!(t.app.world().get::<Selected>(e).is_none());
    send(&mut t, &[release(Hand::Mouse)]);
    // With the trim off, C is ignored: the mode stays the cube's.
    let off = Trim {
        mode: TrimMode::Off,
        ..unit_box()
    };
    send(
        &mut t,
        &[
            act(e, InstanceAction::Select),
            act(e, InstanceAction::SetTrim(off)),
        ],
    );
    point(&mut t, &[down(Hand::Mouse, Vec3::new(1.0, 0.0, 5.0))]);
    assert!(
        handles(&t, e).hovered.is_empty(),
        "no handles with the trim off"
    );
    send(&mut t, &[Intent::Gizmo(GizmoMode::Trim)]);
    send(&mut t, &[act(e, InstanceAction::SetTrim(unit_box()))]);
    send(
        &mut t,
        &[
            down(Hand::Mouse, Vec3::new(1.0, 0.0, 5.0)),
            press(Hand::Mouse),
        ],
    );
    assert!(dragged(&t, e).is_empty(), "C was ignored");
    // C with the trim on takes the handles.
    send(
        &mut t,
        &[
            release(Hand::Mouse),
            act(e, InstanceAction::Select),
            Intent::Gizmo(GizmoMode::Trim),
        ],
    );
    send(
        &mut t,
        &[
            down(Hand::Mouse, Vec3::new(1.0, 0.0, 5.0)),
            press(Hand::Mouse),
        ],
    );
    assert_eq!(dragged(&t, e), [(Hand::Mouse, PLUS_X)]);
}

#[test]
fn i24_desktop_turning_the_trim_off_returns_the_gizmo_to_translate() {
    let (mut t, e) = setup("i24_desktop_off", unit_box());
    send(&mut t, &[Intent::Gizmo(GizmoMode::Trim)]);
    let off = Trim {
        mode: TrimMode::Off,
        ..unit_box()
    };
    send(&mut t, &[act(e, InstanceAction::SetTrim(off))]);
    send(&mut t, &[act(e, InstanceAction::SetTrim(unit_box()))]);
    // The −x face's handle at (−1, 0, 0): with the gizmo back on translate, no trim drag.
    point(
        &mut t,
        &[
            down(Hand::Mouse, Vec3::new(-1.0, 0.0, 5.0)),
            press(Hand::Mouse),
        ],
    );
    assert!(dragged(&t, e).is_empty());
    assert!(handles(&t, e).hovered.is_empty());
}

#[test]
fn i24_xr_a_grab_drag_moves_a_face_by_the_point_on_the_ray() {
    let (mut t, e) = setup("i24_xr_face", unit_box());
    // Hovering the handle hovers it, not the instance.
    point(&mut t, &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0))]);
    assert_eq!(handles(&t, e).hovered, [(Hand::Left, PLUS_X)]);
    assert!(t.app.world().get::<Hovered>(e).is_none());
    send(&mut t, &[grab(Hand::Left)]);
    assert_eq!(dragged(&t, e), [(Hand::Left, PLUS_X)]);
    assert!(
        t.app.world().get::<Grabbed>(e).is_none(),
        "the handle, not the instance"
    );
    // t_h = 5: the hand moved 0.5 along x gives p = (1.5, 0, 0).
    send(&mut t, &[down(Hand::Left, Vec3::new(1.5, 0.0, 5.0))]);
    assert!(close(trim(&t, e).box_max.x, 1.5));
    // Off the axis too: only δ's x counts.
    send(&mut t, &[down(Hand::Left, Vec3::new(1.25, 0.4, 4.0))]);
    let tr = trim(&t, e);
    assert!(close(tr.box_max.x, 1.25), "{tr:?}");
    assert_eq!((tr.box_max.y, tr.box_max.z), (1.0, 1.0));
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Left }]);
    assert!(dragged(&t, e).is_empty());
}

#[test]
fn i24_xr_a_select_drag_moves_the_center() {
    let (mut t, e) = setup("i24_xr_center", wide_sphere());
    send(
        &mut t,
        &[diagonal(Hand::Right, Vec3::ZERO), press(Hand::Right)],
    );
    assert_eq!(dragged(&t, e), [(Hand::Right, TrimHandle::Center)]);
    send(&mut t, &[diagonal(Hand::Right, Vec3::new(0.5, 0.0, -0.25))]);
    let c = trim(&t, e).center;
    assert!(
        close(c.x, 0.5) && close(c.y, 0.0) && close(c.z, -0.25),
        "{c}"
    );
    send(&mut t, &[release(Hand::Right)]);
    assert!(dragged(&t, e).is_empty());
}

#[test]
fn i24_xr_a_radius_drag_stops_at_the_floor() {
    let (mut t, e) = setup(
        "i24_xr_radius",
        Trim {
            radius: 1.0,
            ..wide_sphere()
        },
    );
    let plus_x = TrimHandle::Radius {
        axis: 0,
        positive: true,
    };
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    assert_eq!(dragged(&t, e), [(Hand::Left, plus_x)]);
    send(&mut t, &[down(Hand::Left, Vec3::new(1.5, 0.0, 5.0))]);
    assert!(close(trim(&t, e).radius, 1.5));
    send(&mut t, &[down(Hand::Left, Vec3::new(-3.0, 0.0, 5.0))]);
    assert!(close(trim(&t, e).radius, FLOOR), "{}", trim(&t, e).radius);
}

#[test]
fn i24_xr_hands_take_one_handle_each_and_both_apply() {
    let (mut t, e) = setup("i24_xr_two_hands", unit_box());
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    // A pinch's second action on the dragging hand does nothing.
    send(&mut t, &[press(Hand::Left)]);
    assert_eq!(dragged(&t, e), [(Hand::Left, PLUS_X)]);
    // The right hand cannot take the dragged handle, nor grab the instance.
    send(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(1.0, 0.0, 5.0)),
            grab(Hand::Right),
        ],
    );
    assert_eq!(dragged(&t, e), [(Hand::Left, PLUS_X)]);
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Right }]);
    // It can drag another handle; both drags apply each frame.
    send(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(-1.0, 0.0, 5.0)),
            grab(Hand::Right),
        ],
    );
    assert_eq!(
        dragged(&t, e),
        [(Hand::Left, PLUS_X), (Hand::Right, MINUS_X)]
    );
    send(
        &mut t,
        &[
            down(Hand::Left, Vec3::new(1.25, 0.0, 5.0)),
            down(Hand::Right, Vec3::new(-1.5, 0.0, 5.0)),
        ],
    );
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, 1.25) && close(tr.box_min.x, -1.5),
        "{tr:?}"
    );
}

#[test]
fn i24_xr_no_instance_grab_during_a_drag_and_no_handles_while_grabbed() {
    let (mut t, e) = setup("i24_xr_exclusions", unit_box());
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    // The right ray through the instance's box, away from every handle.
    send(
        &mut t,
        &[diagonal(Hand::Right, Vec3::ZERO), grab(Hand::Right)],
    );
    assert!(
        t.app.world().get::<Grabbed>(e).is_none(),
        "no grab during a drag"
    );
    send(
        &mut t,
        &[
            Intent::GrabEnd { hand: Hand::Right },
            Intent::GrabEnd { hand: Hand::Left },
        ],
    );
    // Grabbed by the right hand: the handles are hidden from the left.
    send(
        &mut t,
        &[diagonal(Hand::Right, Vec3::ZERO), grab(Hand::Right)],
    );
    assert!(t.app.world().get::<Grabbed>(e).is_some());
    point(&mut t, &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0))]);
    assert!(handles(&t, e).hovered.is_empty());
    send(&mut t, &[grab(Hand::Left)]);
    assert!(dragged(&t, e).is_empty());
}

#[test]
fn i24_xr_a_deselect_or_a_trim_change_ends_the_drag() {
    let (mut t, e) = setup("i24_xr_ends", unit_box());
    let start = |t: &mut TestApp| {
        send(
            t,
            &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
        );
        assert_eq!(dragged(t, e), [(Hand::Left, PLUS_X)]);
    };
    start(&mut t);
    send(&mut t, &[act(e, InstanceAction::Deselect)]);
    assert!(dragged(&t, e).is_empty(), "a deselect ends it");
    send(
        &mut t,
        &[
            Intent::GrabEnd { hand: Hand::Left },
            act(e, InstanceAction::Select),
        ],
    );
    start(&mut t);
    // Switched to the sphere: the drag ends, keeping the shape it had reached.
    send(&mut t, &[down(Hand::Left, Vec3::new(1.5, 0.0, 5.0))]);
    let mut sphere = trim(&t, e);
    assert!(close(sphere.box_max.x, 1.5));
    sphere.mode = TrimMode::Sphere;
    send(&mut t, &[act(e, InstanceAction::SetTrim(sphere))]);
    assert!(dragged(&t, e).is_empty(), "a trim change ends it");
    send(&mut t, &[down(Hand::Left, Vec3::new(2.0, 0.0, 5.0))]);
    assert_eq!(trim(&t, e), sphere);
}

const PLUS_Y: TrimHandle = TrimHandle::Face { axis: 1, max: true };

/// Q-379: the left hand drags the +x face and the right hand then the +y face, both
/// moving; t_h = 5 for both.
fn two_faces(name: &str) -> (TestApp, Entity) {
    let (mut t, e) = setup(name, unit_box());
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    send(&mut t, &[down(Hand::Left, Vec3::new(1.25, 0.0, 5.0))]);
    send(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(0.0, 1.0, 5.0)),
            grab(Hand::Right),
        ],
    );
    assert_eq!(
        dragged(&t, e),
        [(Hand::Left, PLUS_X), (Hand::Right, PLUS_Y)]
    );
    send(
        &mut t,
        &[
            down(Hand::Left, Vec3::new(1.5, 0.0, 5.0)),
            down(Hand::Right, Vec3::new(0.0, 1.5, 5.0)),
        ],
    );
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, 1.5) && close(tr.box_max.y, 1.5),
        "{tr:?}"
    );
    (t, e)
}

#[test]
fn i24_q379_the_right_hand_releases_and_its_face_keeps_its_limit() {
    let (mut t, e) = two_faces("i24_q379_right_releases");
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Right }]);
    assert_eq!(dragged(&t, e), [(Hand::Left, PLUS_X)]);
    send(&mut t, &[down(Hand::Left, Vec3::new(1.75, 0.0, 5.0))]);
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.y, 1.5),
        "+y keeps the right hand's value: {tr:?}"
    );
    assert!(
        close(tr.box_max.x, 1.75),
        "+x follows the left hand: {tr:?}"
    );
}

#[test]
fn i24_q379_the_left_hand_releases_first_and_its_face_keeps_its_limit() {
    let (mut t, e) = two_faces("i24_q379_left_releases");
    send(&mut t, &[down(Hand::Left, Vec3::new(1.625, 0.0, 5.0))]);
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Left }]);
    assert_eq!(dragged(&t, e), [(Hand::Right, PLUS_Y)]);
    send(&mut t, &[down(Hand::Right, Vec3::new(0.0, 1.75, 5.0))]);
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, 1.625),
        "+x keeps the left hand's value: {tr:?}"
    );
    assert!(
        close(tr.box_max.y, 1.75),
        "+y follows the right hand: {tr:?}"
    );
}

/// Q-379: the right hand drags the sphere's +x radius handle and the left hand its center;
/// the radius moved to 2.5 and the center to (0.5, 0, −0.25).
fn center_and_radius(name: &str) -> (TestApp, Entity) {
    let (mut t, e) = setup(name, wide_sphere());
    let plus_x = TrimHandle::Radius {
        axis: 0,
        positive: true,
    };
    send(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(2.0, 0.0, 5.0)),
            grab(Hand::Right),
        ],
    );
    send(
        &mut t,
        &[diagonal(Hand::Left, Vec3::ZERO), press(Hand::Left)],
    );
    assert_eq!(
        dragged(&t, e),
        [(Hand::Right, plus_x), (Hand::Left, TrimHandle::Center)]
    );
    send(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(2.5, 0.0, 5.0)),
            diagonal(Hand::Left, Vec3::new(0.5, 0.0, -0.25)),
        ],
    );
    let tr = trim(&t, e);
    assert!(close(tr.radius, 2.5), "{tr:?}");
    assert!(
        close(tr.center.x, 0.5) && close(tr.center.z, -0.25),
        "{tr:?}"
    );
    (t, e)
}

#[test]
fn i24_q379_either_release_keeps_the_other_hands_center_or_radius() {
    // The radius's hand releases: the center moves on, the radius stays.
    let (mut t, e) = center_and_radius("i24_q379_sphere_radius_releases");
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Right }]);
    send(&mut t, &[diagonal(Hand::Left, Vec3::new(1.0, 0.0, -0.5))]);
    let tr = trim(&t, e);
    assert!(close(tr.radius, 2.5), "the radius stays: {tr:?}");
    assert!(
        close(tr.center.x, 1.0) && close(tr.center.y, 0.0) && close(tr.center.z, -0.5),
        "{tr:?}"
    );
    // The center's hand releases: the radius moves on, the center stays.
    let (mut t, e) = center_and_radius("i24_q379_sphere_center_releases");
    send(&mut t, &[release(Hand::Left)]);
    send(&mut t, &[down(Hand::Right, Vec3::new(3.0, 0.0, 5.0))]);
    let tr = trim(&t, e);
    assert!(close(tr.radius, 3.0), "{tr:?}");
    assert!(
        close(tr.center.x, 0.5) && close(tr.center.y, 0.0) && close(tr.center.z, -0.25),
        "the center stays: {tr:?}"
    );
}

fn close_trim(a: Trim, b: Trim) -> bool {
    a.mode == b.mode
        && (a.box_min - b.box_min).abs().max_element() <= TOL
        && (a.box_max - b.box_max).abs().max_element() <= TOL
        && (a.center - b.center).abs().max_element() <= TOL
        && close(a.radius, b.radius)
}

#[test]
fn i24_q379_no_jump_when_a_drag_starts_or_ends() {
    let (mut t, e) = setup("i24_q379_no_jump", unit_box());
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    send(&mut t, &[down(Hand::Left, Vec3::new(1.25, 0.0, 5.0))]);
    // A start, the rays unmoved.
    let before = trim(&t, e);
    point(
        &mut t,
        &[
            down(Hand::Right, Vec3::new(0.0, 1.0, 5.0)),
            grab(Hand::Right),
        ],
    );
    t.frames(1);
    assert_eq!(
        dragged(&t, e),
        [(Hand::Left, PLUS_X), (Hand::Right, PLUS_Y)]
    );
    assert!(close_trim(trim(&t, e), before), "{:?}", trim(&t, e));
    // Both move on, so that neither drag's start holds the other's last value.
    send(
        &mut t,
        &[
            down(Hand::Left, Vec3::new(1.5, 0.0, 5.0)),
            down(Hand::Right, Vec3::new(0.0, 1.5, 5.0)),
        ],
    );
    // An end of each, the rays unmoved.
    for hand in [Hand::Right, Hand::Left] {
        let before = trim(&t, e);
        send(&mut t, &[Intent::GrabEnd { hand }]);
        assert!(!dragged(&t, e).iter().any(|(h, _)| *h == hand));
        assert!(close_trim(trim(&t, e), before), "{:?}", trim(&t, e));
    }
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, 1.5) && close(tr.box_max.y, 1.5),
        "{tr:?}"
    );
}

#[test]
fn i24_q379_one_drag_past_the_opposite_and_back_restores_it() {
    let (mut t, e) = setup("i24_q379_one_drag_back", unit_box());
    send(
        &mut t,
        &[down(Hand::Left, Vec3::new(1.0, 0.0, 5.0)), grab(Hand::Left)],
    );
    send(&mut t, &[down(Hand::Left, Vec3::new(-1.5, 0.0, 5.0))]);
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, -1.5) && close(tr.box_min.x, -1.5),
        "{tr:?}"
    );
    send(&mut t, &[down(Hand::Left, Vec3::new(1.25, 0.0, 5.0))]);
    let tr = trim(&t, e);
    assert!(
        close(tr.box_max.x, 1.25) && close(tr.box_min.x, -1.0),
        "{tr:?}"
    );
}
