// SPDX-License-Identifier: Apache-2.0

//! I-6 (`cpu`, Appendix D.1, §1.7): every row's default key produces its intent through
//! `DesktopInputPlugin`, from injected keyboard and mouse messages; a rebinding in
//! `Settings.desktop.keys` takes effect (Q-152, Q-197). Delete's row asks `ui` for a
//! confirmation (Q-199) and writes no intent; Quit is the window's close action.

use crate::support;

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
        (&[], KeyCode::KeyG, Intent::Gizmo(GizmoMode::Translate)),
        (&[], KeyCode::KeyR, Intent::Gizmo(GizmoMode::Rotate)),
        (&[], KeyCode::KeyT, Intent::Gizmo(GizmoMode::Scale)),
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
        // The rows of several keys (arrows, PageUp and PageDown, the brackets, the
        // bookmarks) are pressed key by key in
        // `i06_every_key_of_the_multi_key_rows_produces_its_intent`.
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

// Q-198
#[test]
fn i06_every_key_of_the_multi_key_rows_produces_its_intent() {
    // Appendix D.1's rows that list several keys, each key pressed on its own: the
    // arrows (1 cm, 10 cm with Shift) and PageUp and PageDown in the stage's axes, `[`
    // and `]` (5°), Ctrl+n and Alt+n for n = 1 … 9 (§1.7, Q-198).
    let mut t = windowed("i06_multi_key_rows");
    let e = load_p1(&mut t);
    t.app.world_mut().entity_mut(e).insert(Selected);
    t.frames(1);
    take(&mut t);
    let nudge = |delta| Intent::Nudge { instance: e, delta };
    let mut rows: Vec<(&[KeyCode], KeyCode, Intent)> = Vec::new();
    let arrows = [
        (KeyCode::ArrowLeft, Vec3::NEG_X),
        (KeyCode::ArrowRight, Vec3::X),
        (KeyCode::ArrowUp, Vec3::NEG_Z),
        (KeyCode::ArrowDown, Vec3::Z),
    ];
    for (k, d) in arrows {
        rows.push((&[], k, nudge(d * 0.01)));
    }
    rows.push((&[], KeyCode::PageUp, nudge(Vec3::Y * 0.01)));
    rows.push((&[], KeyCode::PageDown, nudge(Vec3::NEG_Y * 0.01)));
    for (k, yaw_deg) in [(KeyCode::BracketLeft, 5.0), (KeyCode::BracketRight, -5.0)] {
        rows.push((
            &[],
            k,
            Intent::Turn {
                instance: e,
                yaw_deg,
            },
        ));
    }
    let digits = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (n, k) in (1u8..=9).zip(digits) {
        rows.push((&[CTRL], k, cam(CameraCommand::SaveBookmark(n))));
        rows.push((&[ALT], k, cam(CameraCommand::RecallBookmark(n))));
    }
    for (mods, k, want) in rows {
        let got = no_pointer(chord(&mut t, mods, k));
        assert!(
            has(&got, want),
            "{mods:?} {k:?}: want {want:?}, got {got:?}"
        );
    }
    // With Shift, 10 cm: 10 × 0.01 m in f32 is not exactly 0.1, so the step is held to
    // I-8's 10⁻⁶.
    for (k, d) in arrows {
        let got = no_pointer(chord(&mut t, &[SHIFT], k));
        assert!(
            got.iter().any(|i| matches!(i,
                Intent::Nudge { instance, delta }
                    if *instance == e && (*delta - d * 0.1).length() < 1e-6)),
            "Shift {k:?}: want a nudge of {}, got {got:?}",
            d * 0.1
        );
    }
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

// Q-197
#[test]
fn ctrl_shift_s_saves_the_scene_without_moving_back() {
    // Appendix D.1: Ctrl+Shift+S is "save scene"; S alone (with Shift or Ctrl for speed)
    // moves back. The chord's frame moves nothing.
    let mut t = support::desktop::windowed("i06_chord_no_move");
    t.frames(1);
    let seen = support::desktop::chord(
        &mut t,
        &[KeyCode::ControlLeft, KeyCode::ShiftLeft],
        KeyCode::KeyS,
    );
    assert!(seen.contains(&Intent::Ui(UiCommand::SaveScene)), "{seen:?}");
    assert!(
        !seen
            .iter()
            .any(|i| matches!(i, Intent::Camera(CameraCommand::Move(_)))),
        "{seen:?}"
    );
    // Shift+S still moves back, fast.
    let seen = support::desktop::chord(&mut t, &[KeyCode::ShiftLeft], KeyCode::KeyS);
    assert!(
        seen.iter()
            .any(|i| matches!(i, Intent::Camera(CameraCommand::Move(v)) if v.z > 0.0)),
        "{seen:?}"
    );
}
