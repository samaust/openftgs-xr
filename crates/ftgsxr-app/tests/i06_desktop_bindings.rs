// SPDX-License-Identifier: Apache-2.0

//! I-6 (`cpu`, Appendix D.1, §1.7): every row's default key produces its intent through
//! `DesktopInputPlugin`, from injected keyboard and mouse messages; a rebinding in
//! `Settings.desktop.keys` takes effect (Q-152, Q-197). Delete's row asks `ui` for a
//! confirmation (Q-199) and writes no intent; Quit is the window's close action.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{
    CameraCommand, GizmoMode, InstanceAction, Intent, OverlayToggle, PlaybackCommand, Selected,
    SettingsRes, UiCommand,
};
use ftgsxr_render::{Colormap, RenderMode};
use support::TestApp;
use support::desktop::{button, chord, load_p1, motion, take, wheel, windowed};

const CTRL: KeyCode = KeyCode::ControlLeft;
const SHIFT: KeyCode = KeyCode::ShiftLeft;
const ALT: KeyCode = KeyCode::AltLeft;

fn cam(c: CameraCommand) -> Intent {
    Intent::Camera(c)
}

fn has(got: &[Intent], want: Intent) -> bool {
    got.contains(&want)
}

/// The intents other than the mouse's pointer ray.
fn no_pointer(v: Vec<Intent>) -> Vec<Intent> {
    v.into_iter()
        .filter(|i| !matches!(i, Intent::Pointer { .. }))
        .collect()
}

#[test]
fn i06_every_key_row_produces_its_intent() {
    let mut t = windowed("i06_keys");
    let e = load_p1(&mut t);
    t.app.world_mut().entity_mut(e).insert(Selected);
    t.frames(1);
    take(&mut t);
    let inst = |action| Intent::Instance {
        instance: e,
        action,
    };
    let rows: Vec<(&[KeyCode], KeyCode, Intent)> = vec![
        (&[], KeyCode::KeyW, cam(CameraCommand::Move(Vec3::NEG_Z))),
        (&[], KeyCode::KeyA, cam(CameraCommand::Move(Vec3::NEG_X))),
        (&[], KeyCode::KeyS, cam(CameraCommand::Move(Vec3::Z))),
        (&[], KeyCode::KeyD, cam(CameraCommand::Move(Vec3::X))),
        (&[], KeyCode::KeyE, cam(CameraCommand::Move(Vec3::Y))),
        (&[], KeyCode::KeyQ, cam(CameraCommand::Move(Vec3::NEG_Y))),
        (
            &[SHIFT],
            KeyCode::KeyW,
            cam(CameraCommand::Move(Vec3::NEG_Z * 4.0)),
        ),
        (
            &[CTRL],
            KeyCode::KeyW,
            cam(CameraCommand::Move(Vec3::NEG_Z * 0.25)),
        ),
        (&[], KeyCode::KeyO, cam(CameraCommand::ToggleOrbit)),
        (&[], KeyCode::KeyF, cam(CameraCommand::Frame)),
        (&[], KeyCode::Home, cam(CameraCommand::ResetView)),
        (
            &[CTRL],
            KeyCode::Digit1,
            cam(CameraCommand::SaveBookmark(1)),
        ),
        (
            &[CTRL],
            KeyCode::Digit9,
            cam(CameraCommand::SaveBookmark(9)),
        ),
        (
            &[ALT],
            KeyCode::Digit3,
            cam(CameraCommand::RecallBookmark(3)),
        ),
        (&[], KeyCode::KeyG, Intent::Gizmo(GizmoMode::Translate)),
        (&[], KeyCode::KeyR, Intent::Gizmo(GizmoMode::Rotate)),
        (&[], KeyCode::KeyT, Intent::Gizmo(GizmoMode::Scale)),
        (
            &[],
            KeyCode::ArrowLeft,
            Intent::Nudge {
                instance: e,
                delta: Vec3::new(-0.01, 0.0, 0.0),
            },
        ),
        (
            &[],
            KeyCode::PageUp,
            Intent::Nudge {
                instance: e,
                delta: Vec3::new(0.0, 0.01, 0.0),
            },
        ),
        (
            &[],
            KeyCode::BracketLeft,
            Intent::Turn {
                instance: e,
                yaw_deg: 5.0,
            },
        ),
        (&[CTRL], KeyCode::KeyF, inst(InstanceAction::Fit)),
        (&[CTRL], KeyCode::KeyP, inst(InstanceAction::SavePose)),
        (&[CTRL], KeyCode::Backspace, inst(InstanceAction::ResetPose)),
        (&[CTRL], KeyCode::KeyO, Intent::Ui(UiCommand::OpenFiles)),
        (
            &[CTRL, SHIFT],
            KeyCode::KeyS,
            Intent::Ui(UiCommand::SaveScene),
        ),
        (&[CTRL], KeyCode::KeyL, Intent::Ui(UiCommand::LoadScene)),
        (
            &[],
            KeyCode::Space,
            Intent::Playback(PlaybackCommand::Toggle),
        ),
        (
            &[],
            KeyCode::Comma,
            Intent::Playback(PlaybackCommand::StepFrames(-1)),
        ),
        (
            &[],
            KeyCode::Period,
            Intent::Playback(PlaybackCommand::StepFrames(1)),
        ),
        // §1.6's presets: from the default 1 down to 0.5, then up again to 1.
        (
            &[SHIFT],
            KeyCode::Comma,
            Intent::Playback(PlaybackCommand::Speed(0.5)),
        ),
        (
            &[SHIFT],
            KeyCode::Period,
            Intent::Playback(PlaybackCommand::Speed(1.0)),
        ),
        (&[], KeyCode::KeyL, Intent::Ui(UiCommand::CycleLoop)),
        (
            &[CTRL],
            KeyCode::Home,
            Intent::Playback(PlaybackCommand::Seek(0.0)),
        ),
        (&[], KeyCode::KeyB, Intent::Overlay(OverlayToggle::Boxes)),
        (
            &[],
            KeyCode::KeyV,
            Intent::Overlay(OverlayToggle::Velocities),
        ),
        (&[], KeyCode::F1, Intent::Ui(UiCommand::Help)),
        (&[], KeyCode::F2, Intent::Ui(UiCommand::HideUi)),
        (&[], KeyCode::F3, Intent::Ui(UiCommand::Hud)),
        (&[], KeyCode::F11, Intent::Ui(UiCommand::Fullscreen)),
        (&[], KeyCode::F12, Intent::Screenshot),
        (
            &[],
            KeyCode::KeyM,
            inst(InstanceAction::SetMode(RenderMode::Centers {
                pixel_radius: 3.0,
                colormap: Colormap::Dc,
            })),
        ),
        // Esc last: it deselects.
        (&[], KeyCode::Escape, inst(InstanceAction::Deselect)),
    ];
    for (mods, k, want) in rows {
        let got = no_pointer(chord(&mut t, mods, k));
        assert!(
            has(&got, want),
            "{mods:?} {k:?}: want {want:?}, got {got:?}"
        );
    }
    // Delete asks `ui` for a confirmation and unloads nothing by itself (Q-199).
    let got = no_pointer(chord(&mut t, &[], KeyCode::Delete));
    assert!(!has(&got, inst(InstanceAction::Unload)), "{got:?}");
    // Ctrl+End seeks to the clock's end.
    let got = no_pointer(chord(&mut t, &[CTRL], KeyCode::End));
    assert!(
        got.iter()
            .any(|i| matches!(i, Intent::Playback(PlaybackCommand::Seek(_)))),
        "{got:?}"
    );
}

#[test]
fn i06_a_chord_does_not_fire_its_unmodified_key() {
    let mut t = windowed("i06_exact");
    let got = no_pointer(chord(&mut t, &[SHIFT], KeyCode::Comma));
    assert!(
        !got.contains(&Intent::Playback(PlaybackCommand::StepFrames(-1))),
        "{got:?}"
    );
    let got = no_pointer(chord(&mut t, &[CTRL], KeyCode::Home));
    assert!(!got.contains(&cam(CameraCommand::ResetView)), "{got:?}");
}

#[test]
fn i06_mouse_rows_produce_their_intents() {
    let mut t = windowed("i06_mouse");
    // Look: the right button held with motion (fly).
    button(&mut t, MouseButton::Right, true);
    motion(&mut t, Vec2::new(10.0, -4.0));
    t.frames(1);
    button(&mut t, MouseButton::Right, false);
    t.frames(1);
    assert!(has(
        &take(&mut t),
        cam(CameraCommand::Look(Vec2::new(10.0, -4.0)))
    ));
    // The wheel: the base speed in fly.
    wheel(&mut t, 1.0);
    t.frames(1);
    assert!(has(&take(&mut t), cam(CameraCommand::Speed(1.0))));
    // Orbit: zoom, left drag, middle drag.
    chord(&mut t, &[], KeyCode::KeyO);
    t.frames(1);
    wheel(&mut t, -2.0);
    t.frames(1);
    assert!(has(&take(&mut t), cam(CameraCommand::Zoom(-2.0))));
    button(&mut t, MouseButton::Left, true);
    t.frames(1);
    motion(&mut t, Vec2::new(3.0, 1.0));
    t.frames(1);
    button(&mut t, MouseButton::Left, false);
    t.frames(1);
    assert!(has(
        &take(&mut t),
        cam(CameraCommand::Orbit(Vec2::new(3.0, 1.0)))
    ));
    button(&mut t, MouseButton::Middle, true);
    t.frames(1);
    motion(&mut t, Vec2::new(-2.0, 5.0));
    t.frames(1);
    button(&mut t, MouseButton::Middle, false);
    t.frames(1);
    assert!(has(
        &take(&mut t),
        cam(CameraCommand::Pan(Vec2::new(-2.0, 5.0)))
    ));
    // A double click frames.
    for _ in 0..2 {
        button(&mut t, MouseButton::Left, true);
        t.frames(1);
        button(&mut t, MouseButton::Left, false);
        t.frames(1);
    }
    let got = take(&mut t);
    assert!(has(&got, cam(CameraCommand::Frame)), "{got:?}");
}

#[test]
fn i06_a_rebinding_in_the_settings_takes_effect() {
    let mut t: TestApp = windowed("i06_rebind");
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .desktop
        .keys
        .insert("play_pause".into(), "KeyP".into());
    t.frames(1);
    let toggle = Intent::Playback(PlaybackCommand::Toggle);
    assert!(has(&chord(&mut t, &[], KeyCode::KeyP), toggle));
    assert!(!has(&chord(&mut t, &[], KeyCode::Space), toggle));
}
