// SPDX-License-Identifier: Apache-2.0

//! `KeyMap`'s names through `Settings.desktop.keys` (`cpu`, #284; source of truth
//! Appendix D.1 and Q-152, Q-197): every default's action and key name, written to
//! a settings file and read back, rebuilds the same binding; an unknown action or key
//! keeps the default with a warning.

use std::collections::BTreeMap;

use ftgsxr_app::{KeyMap, Settings, load_settings};
use ftgsxr_io::save_settings;

/// Appendix D.1's rows under Q-152's names, with Q-197's key spelling.
const D1: &[(&str, &str)] = &[
    ("move_forward", "KeyW"),
    ("move_left", "KeyA"),
    ("move_back", "KeyS"),
    ("move_right", "KeyD"),
    ("move_up", "KeyE"),
    ("move_down", "KeyQ"),
    ("toggle_fly_orbit", "KeyO"),
    ("frame_selected", "KeyF"),
    ("reset_view", "Home"),
    ("save_bookmark_n", "Ctrl+Digitn"),
    ("recall_bookmark_n", "Alt+Digitn"),
    ("deselect", "Escape"),
    ("gizmo_translate", "KeyG"),
    ("gizmo_rotate", "KeyR"),
    ("gizmo_scale", "KeyT"),
    ("gizmo_trim", "KeyC"), // Q-377
    ("nudge_left", "ArrowLeft"),
    ("nudge_right", "ArrowRight"),
    ("nudge_forward", "ArrowUp"),
    ("nudge_back", "ArrowDown"),
    ("nudge_up", "PageUp"),
    ("nudge_down", "PageDown"),
    ("turn_left", "BracketLeft"),
    ("turn_right", "BracketRight"),
    ("fit", "Ctrl+KeyF"),
    ("save_pose", "Ctrl+KeyP"),
    ("reset_pose", "Ctrl+Backspace"),
    ("unload", "Delete"),
    ("open_files", "Ctrl+KeyO"),
    ("save_scene", "Ctrl+Shift+KeyS"),
    ("load_scene", "Ctrl+KeyL"),
    ("play_pause", "Space"),
    ("step_back", "Comma"),
    ("step_forward", "Period"),
    ("speed_down", "Shift+Comma"),
    ("speed_up", "Shift+Period"),
    ("cycle_loop", "KeyL"),
    ("clock_start", "Ctrl+Home"),
    ("clock_end", "Ctrl+End"),
    ("cycle_mode", "KeyM"),
    ("toggle_boxes", "KeyB"),
    ("toggle_velocities", "KeyV"),
    ("help", "F1"),
    ("hide_ui", "F2"),
    ("hud", "F3"),
    ("fullscreen", "F11"),
    ("screenshot", "F12"),
];

#[test]
fn defaults_are_appendix_d1() {
    let m = KeyMap::default();
    let got: Vec<(&str, String)> = KeyMap::actions()
        .iter()
        .map(|a| (*a, m.key(a).unwrap()))
        .collect();
    let want: Vec<(&str, String)> = D1.iter().map(|(a, k)| (*a, k.to_string())).collect();
    assert_eq!(got, want);
}

#[test]
fn every_default_round_trips_through_settings_desktop_keys() {
    let dir = std::env::temp_dir().join(format!("ftgsxr-keymap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("settings.toml");
    for (action, key) in D1 {
        let mut s = Settings::default();
        s.desktop.keys = BTreeMap::from([(action.to_string(), key.to_string())]);
        save_settings(&path, &s).unwrap();
        let (back, warnings) = load_settings(&path).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let (m, w) = KeyMap::from_keys(&back.desktop.keys);
        assert!(w.is_empty(), "{action}: {w:?}");
        assert_eq!(m.key(action).as_deref(), Some(*key), "{action}");
    }
    // A rebinding to another action's key, and a chord with every modifier.
    let keys = BTreeMap::from([
        ("play_pause".to_string(), "KeyP".to_string()),
        ("help".to_string(), "Ctrl+Shift+Alt+KeyH".to_string()),
    ]);
    let (m, w) = KeyMap::from_keys(&keys);
    assert!(w.is_empty(), "{w:?}");
    assert_eq!(m.key("play_pause").as_deref(), Some("KeyP"));
    assert_eq!(m.key("help").as_deref(), Some("Ctrl+Shift+Alt+KeyH"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_unknown_action_or_key_is_a_warning_and_the_default() {
    let keys = BTreeMap::from([
        ("fly_to_the_moon".to_string(), "KeyX".to_string()),
        ("play_pause".to_string(), "NoSuchKey".to_string()),
        ("help".to_string(), "Hyper+F1".to_string()),
    ]);
    let (m, w) = KeyMap::from_keys(&keys);
    assert_eq!(w.len(), 3, "{w:?}");
    assert_eq!(m.key("play_pause").as_deref(), Some("Space"));
    assert_eq!(m.key("help").as_deref(), Some("F1"));
    assert_eq!(m.key("fly_to_the_moon"), None);
}
