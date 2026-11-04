// SPDX-License-Identifier: Apache-2.0

//! The desktop input's test support (#284): a windowed `TestApp` whose `Update` collects
//! every `Intent`, and Bevy's keyboard and mouse messages injected as winit would write
//! them, so `ButtonInput` and the accumulated motion and scroll are Bevy's own.

use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput, NativeKey};
use bevy::input::mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use ftgsxr_app::{Intent, LibraryCommand, LoadedModel, ModelRef};
use ftgsxr_testkit::fixtures_dir;

use super::{Setup, TestApp};

/// The intents written since the last `take`.
#[derive(Resource, Default)]
pub struct Seen(pub Vec<Intent>);

fn collect(mut r: MessageReader<Intent>, mut seen: ResMut<Seen>) {
    seen.0.extend(r.read().copied());
}

/// A CPU app with a primary window (1280 × 720 logical pixels, so the desktop camera
/// exists, Q-194) and the intent collector.
#[track_caller]
pub fn windowed(name: &str) -> TestApp {
    let mut t = TestApp::with(name, Setup::default(), |app| {
        app.world_mut().spawn((Window::default(), PrimaryWindow));
        app.init_resource::<Seen>().add_systems(Update, collect);
    });
    t.frames(2);
    take(&mut t);
    t
}

/// The intents collected since the last call.
pub fn take(t: &mut TestApp) -> Vec<Intent> {
    std::mem::take(&mut t.app.world_mut().resource_mut::<Seen>().0)
}

fn window(t: &mut TestApp) -> Entity {
    let w = t.app.world_mut();
    w.query_filtered::<Entity, With<PrimaryWindow>>()
        .single(w)
        .unwrap()
}

/// Writes a key's press or release.
pub fn key(t: &mut TestApp, k: KeyCode, down: bool) {
    let window = window(t);
    t.app.world_mut().write_message(KeyboardInput {
        key_code: k,
        logical_key: Key::Unidentified(NativeKey::Unidentified),
        state: if down {
            ButtonState::Pressed
        } else {
            ButtonState::Released
        },
        text: None,
        repeat: false,
        window,
    });
}

/// Presses `mods` then `k` in one frame, releases them all the next, and returns the
/// intents of both frames.
pub fn chord(t: &mut TestApp, mods: &[KeyCode], k: KeyCode) -> Vec<Intent> {
    for m in mods {
        key(t, *m, true);
    }
    key(t, k, true);
    t.frames(1);
    for m in mods {
        key(t, *m, false);
    }
    key(t, k, false);
    t.frames(1);
    take(t)
}

/// Writes a mouse button's press or release.
pub fn button(t: &mut TestApp, b: MouseButton, down: bool) {
    let window = window(t);
    t.app.world_mut().write_message(MouseButtonInput {
        button: b,
        state: if down {
            ButtonState::Pressed
        } else {
            ButtonState::Released
        },
        window,
    });
}

/// Writes a mouse motion.
pub fn motion(t: &mut TestApp, d: Vec2) {
    t.app.world_mut().write_message(MouseMotion { delta: d });
}

/// Writes a wheel event of `y` lines.
pub fn wheel(t: &mut TestApp, y: f32) {
    let window = window(t);
    t.app.world_mut().write_message(MouseWheel {
        unit: MouseScrollUnit::Line,
        x: 0.0,
        y,
        window,
        phase: TouchPhase::Moved,
    });
}

/// Puts the cursor at `p` logical pixels from the window's top left.
pub fn cursor(t: &mut TestApp, p: Option<Vec2>) {
    let w = t.app.world_mut();
    let mut q = w.query_filtered::<&mut Window, With<PrimaryWindow>>();
    q.single_mut(w).unwrap().set_cursor_position(p);
}

/// Loads P1 and returns its instance.
pub fn load_p1(t: &mut TestApp) -> Entity {
    let path = fixtures_dir().join("probes/P1.safetensors");
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "P1 loads"
    );
    t.frames(1);
    take(t);
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    q.single(t.app.world()).unwrap()
}
