// SPDX-License-Identifier: Apache-2.0

//! U-29: `ftgsxr_io::settings` (§1.11, D-23): defaults; a TOML round trip; unknown keys
//! ignored with a warning; a wrong-type key takes its default with a warning; major
//! version 2 is set aside and defaults load; the write is atomic (a temporary file
//! renamed). Values without a §1 default follow Q-49 to Q-51.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ftgsxr_core::LoopMode;
use ftgsxr_io::{
    AudioSettings, CacheSettings, DesktopSettings, DisplaySettings, EnvironmentSettings,
    GeneralSettings, ListingFlavor, LocomotionSettings, ManualLighting, OverlaySettings,
    PlaybackSettings, Settings, SoundSwitches, SourceSpec, XrSettings, load_settings,
    save_settings,
};

/// A fresh, empty directory of this test binary's own under the target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("u29")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the test directory");
    dir
}

/// The names of the files in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("list the test directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// U-29 defaults: every key carries the default §1 gives (§1.4, §1.6–§1.13, E12–E14),
/// and Q-50's where §1 gives none; a missing file loads them with no warning.
#[test]
fn u29_defaults_are_section_1s() {
    let s = Settings::default();
    let tier = if cfg!(target_os = "android") {
        "quest-balanced"
    } else {
        "desktop"
    };
    assert_eq!(
        s.general,
        GeneralSettings {
            tier: tier.into(),
            governor: true,
            ui_scale: 1.0,
            log_level: "info".into(),
        }
    );
    assert!(s.sources.is_empty());
    assert_eq!(
        s.playback,
        PlaybackSettings {
            speed: 1.0,
            loop_mode: LoopMode::Loop,
        }
    );
    assert_eq!(
        s.display,
        DisplaySettings {
            mode: "splats".into(),
            centers_radius: 3.0,
            colormap: "dc".into(),
            grid: true,
            overlays: OverlaySettings {
                boxes: false,
                velocities: false,
                velocity_every: 64,
                time_tint: false,
            },
        }
    );
    assert_eq!(
        s.environment,
        EnvironmentSettings {
            mode: "vr".into(),
            background: [0.1, 0.1, 0.1],
            passthrough_brightness: 0.0,
            passthrough_contrast: 0.0,
            passthrough_saturation: 0.0,
            occlusion: true,
            occlusion_soft: true,
            occlusion_delta_m: 0.05,
            hand_removal: false,
            lighting_mode: "off".into(),
            lighting_manual: ManualLighting {
                exposure_ev: 0.0,
                temperature: 0.0,
                tint: 0.0,
            },
            lighting_gamma: 0.5,
            lighting_reference: None,
            contact_shadows: true,
        }
    );
    assert_eq!(
        s.locomotion,
        LocomotionSettings {
            speed_mps: 2.0,
            snap_turn: true,
            snap_deg: 45.0,
            smooth_dps: 60.0,
            vignette: true,
            teleport: false,
            height_offset_m: 0.0,
            left_handed: false,
            reduced_motion: false,
        }
    );
    assert_eq!(
        s.audio,
        AudioSettings {
            muted: false,
            master: 100.0,
            ui: 100.0,
            soundtrack: 100.0,
            spatial: true,
            sounds: SoundSwitches {
                hover: true,
                select: true,
                grab: true,
                snap_turn: true,
                click: true,
                error: true,
                confirm: true,
            },
        }
    );
    assert_eq!(
        s.desktop,
        DesktopSettings {
            fov_deg: 60.0,
            fly_speed_mps: 2.0,
            keys: BTreeMap::new(),
        }
    );
    assert_eq!(
        s.xr,
        XrSettings {
            refresh_rate: None,
            foveation: None,
            render_scale: None,
            reference_space: vec!["stage".into(), "local-floor".into(), "local".into()],
        }
    );
    assert_eq!(
        s.cache,
        CacheSettings {
            max_bytes: 20 * 1024 * 1024 * 1024,
        }
    );

    let dir = scratch("defaults");
    let (loaded, warnings) = load_settings(&dir.join("settings.toml")).expect("missing → defaults");
    assert_eq!(loaded, s);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(
        names(&dir).is_empty(),
        "loading a missing file writes nothing"
    );
}

/// Settings with every value away from its default: every `Option` set, sources of each
/// kind and flavor, a rebinding.
fn non_default() -> Settings {
    let mut keys = BTreeMap::new();
    keys.insert("play_pause".to_string(), "P".to_string());
    keys.insert("screenshot".to_string(), "Ctrl+F12".to_string());
    Settings {
        general: GeneralSettings {
            tier: "quest-low".into(),
            governor: false,
            ui_scale: 1.25,
            log_level: "debug".into(),
        },
        sources: vec![
            SourceSpec {
                kind: "dir".into(),
                location: "/data/models".into(),
                name: "Local".into(),
                flavor: None,
            },
            SourceSpec {
                kind: "http".into(),
                location: "https://example.org/models/".into(),
                name: "Server".into(),
                flavor: Some(ListingFlavor::NginxJson),
            },
            SourceSpec {
                kind: "http".into(),
                location: "http://192.168.1.2:8000/".into(),
                name: "Laptop".into(),
                flavor: Some(ListingFlavor::Auto),
            },
            SourceSpec {
                kind: "http".into(),
                location: "http://nas.local/a/".into(),
                name: "NAS".into(),
                flavor: Some(ListingFlavor::Html),
            },
            SourceSpec {
                kind: "http".into(),
                location: "http://nas.local/b/".into(),
                name: "NAS index".into(),
                flavor: Some(ListingFlavor::Index),
            },
            SourceSpec {
                kind: "documents".into(),
                location: "content://com.android.externalstorage.documents/tree/primary%3AModels"
                    .into(),
                name: "Picked".into(),
                flavor: None,
            },
        ],
        playback: PlaybackSettings {
            speed: -0.25,
            loop_mode: LoopMode::PingPong,
        },
        display: DisplaySettings {
            mode: "both".into(),
            centers_radius: 5.5,
            colormap: "time".into(),
            grid: false,
            overlays: OverlaySettings {
                boxes: true,
                velocities: true,
                velocity_every: 16,
                time_tint: true,
            },
        },
        environment: EnvironmentSettings {
            mode: "passthrough".into(),
            background: [0.2, 0.3, 0.4],
            passthrough_brightness: -0.1,
            passthrough_contrast: 0.3,
            passthrough_saturation: 0.7,
            occlusion: false,
            occlusion_soft: false,
            occlusion_delta_m: 0.12,
            hand_removal: true,
            lighting_mode: "auto-local".into(),
            lighting_manual: ManualLighting {
                exposure_ev: -1.5,
                temperature: 0.25,
                tint: -0.75,
            },
            lighting_gamma: 0.8,
            lighting_reference: Some([0.4, 0.5, 0.45, 0.3]),
            contact_shadows: false,
        },
        locomotion: LocomotionSettings {
            speed_mps: 3.5,
            snap_turn: false,
            snap_deg: 90.0,
            smooth_dps: 90.0,
            vignette: false,
            teleport: true,
            height_offset_m: -0.3,
            left_handed: true,
            reduced_motion: true,
        },
        audio: AudioSettings {
            muted: true,
            master: 80.0,
            ui: 35.5,
            soundtrack: 0.0,
            spatial: false,
            sounds: SoundSwitches {
                hover: false,
                select: false,
                grab: false,
                snap_turn: false,
                click: false,
                error: false,
                confirm: false,
            },
        },
        desktop: DesktopSettings {
            fov_deg: 75.0,
            fly_speed_mps: 0.1,
            keys,
        },
        xr: XrSettings {
            refresh_rate: Some(90.0),
            foveation: Some("dynamic".into()),
            render_scale: Some(0.8),
            reference_space: vec!["local-floor".into(), "local".into()],
        },
        cache: CacheSettings {
            max_bytes: 5_000_000_123,
        },
    }
}

/// U-29 round trip: a save then a load gives the same settings, with no warning.
#[test]
fn u29_toml_round_trip() {
    let dir = scratch("round_trip");
    let path = dir.join("settings.toml");
    for settings in [non_default(), Settings::default()] {
        save_settings(&path, &settings).expect("save");
        let (loaded, warnings) = load_settings(&path).expect("load");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(loaded, settings);
    }
    let text = std::fs::read_to_string(&path).expect("read back");
    let first = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default();
    assert!(text.contains("format = \"ftgsxr-settings\""), "{text}");
    assert!(text.contains("version = 1"), "{text}");
    assert!(
        !first.starts_with('['),
        "format and version come first: {text}"
    );
}

/// Writes `text` as `settings.toml` in a fresh directory `name` and loads it.
fn load_text(name: &str, text: &str) -> (PathBuf, Settings, Vec<String>) {
    let dir = scratch(name);
    let path = dir.join("settings.toml");
    std::fs::write(&path, text).expect("write the test file");
    let (settings, warnings) = load_settings(&path).expect("load");
    (dir, settings, warnings)
}

/// U-29 unknown keys: ignored with a warning naming each one, at the top level, in a
/// table, in a sub-table and in a source entry; the known keys beside them still load.
#[test]
fn u29_unknown_keys_are_ignored_with_a_warning() {
    let (_, s, warnings) = load_text(
        "unknown_keys",
        r#"
format = "ftgsxr-settings"
version = 1
colour_scheme = "dark"

[general]
ui_scale = 1.5
favourite = 3

[audio.sounds]
hover = false
whoosh = true

[[sources]]
kind = "dir"
location = "/m"
name = "M"
icon = "folder"

[telemetry]
enabled = false
"#,
    );
    let mut expected = Settings::default();
    expected.general.ui_scale = 1.5;
    expected.audio.sounds.hover = false;
    expected.sources.push(SourceSpec {
        kind: "dir".into(),
        location: "/m".into(),
        name: "M".into(),
        flavor: None,
    });
    assert_eq!(s, expected);
    assert_eq!(warnings.len(), 5, "{warnings:?}");
    for key in [
        "`colour_scheme`",
        "`general.favourite`",
        "`audio.sounds.whoosh`",
        "`sources[0].icon`",
        "`telemetry`",
    ] {
        assert!(
            warnings
                .iter()
                .any(|w| w.contains(key) && w.contains("unknown")),
            "no warning names {key}: {warnings:?}"
        );
    }
}

/// U-29 wrong types: a key of the wrong type takes its default with a warning naming it,
/// as does a table that is not a table and a value outside its range or spelling
/// (Q-50); the other keys of the same table still load; an integer is a number.
#[test]
fn u29_wrong_type_keys_take_their_default_with_a_warning() {
    let (_, s, warnings) = load_text(
        "wrong_type",
        r#"
format = "ftgsxr-settings"
version = 1
locomotion = "fast"

[general]
ui_scale = "big"
governor = 0
tier = "ultra"
log_level = "debug"

[display]
centers_radius = 6
background = [0.5, 0.5]
grid = false

[audio]
master = 150
ui = 40

[[sources]]
kind = "ftp"
location = "ftp://example.org/"

[[sources]]
kind = "http"
location = "http://example.org/"
flavor = "xml"

[desktop.keys]
play_pause = 32
help = "F1"
"#,
    );
    let mut expected = Settings::default();
    expected.general.log_level = "debug".into();
    expected.display.centers_radius = 6.0;
    expected.display.grid = false;
    expected.audio.ui = 40.0;
    expected.sources.push(SourceSpec {
        kind: "http".into(),
        location: "http://example.org/".into(),
        name: "http://example.org/".into(),
        flavor: None,
    });
    expected.desktop.keys.insert("help".into(), "F1".into());
    assert_eq!(s, expected);
    for key in [
        "`locomotion`",
        "`general.ui_scale`",
        "`general.governor`",
        "`general.tier`",
        "`display.background`",
        "`audio.master`",
        "`sources[0].kind`",
        "`sources[1].flavor`",
        "`desktop.keys.play_pause`",
    ] {
        assert!(
            warnings.iter().any(|w| w.contains(key)),
            "no warning names {key}: {warnings:?}"
        );
    }
    assert!(
        !warnings.iter().any(|w| w.contains("unknown")),
        "{warnings:?}"
    );
}

/// Q-49: a `[playback] loop` outside Q-50's spellings takes the default
/// `LoopMode::Loop`, with one warning naming the key.
#[test]
fn u29_an_unknown_loop_spelling_takes_the_default_with_a_warning() {
    let (_, s, warnings) = load_text(
        "loop_sideways",
        "format = \"ftgsxr-settings\"\nversion = 1\n\n[playback]\nloop = \"sideways\"\n",
    );
    assert_eq!(s.playback.loop_mode, LoopMode::Loop);
    assert_eq!(s, Settings::default());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("`playback.loop`"), "{warnings:?}");
}

/// U-29 speed (Q-50, E9): any finite speed in \[−4, 4\] other than 0 loads as given with
/// no warning, not only §1.6's presets.
#[test]
fn u29_speed_in_e9s_range_loads_as_given() {
    for (text, speed) in [("1.5", 1.5), ("-3.7", -3.7), ("4", 4.0)] {
        let (_, s, warnings) = load_text(
            "speed_in_range",
            &format!("format = \"ftgsxr-settings\"\nversion = 1\n[playback]\nspeed = {text}\n"),
        );
        assert_eq!(s.playback.speed, speed, "speed = {text}");
        assert!(warnings.is_empty(), "speed = {text}: {warnings:?}");
    }
}

/// U-29 speed (Q-50, E9): 0, a speed outside \[−4, 4\], a non-finite speed and a
/// non-number warn naming `speed` and load the default 1.
#[test]
fn u29_speed_outside_e9s_range_takes_the_default_with_a_warning() {
    for text in ["0", "4.5", "-5", "nan", "inf", "\"fast\""] {
        let (_, s, warnings) = load_text(
            "speed_out_of_range",
            &format!("format = \"ftgsxr-settings\"\nversion = 1\n[playback]\nspeed = {text}\n"),
        );
        assert_eq!(s.playback.speed, 1.0, "speed = {text}");
        assert_eq!(warnings.len(), 1, "speed = {text}: {warnings:?}");
        assert!(
            warnings[0].contains("`playback.speed`"),
            "speed = {text}: {warnings:?}"
        );
    }
}

/// Today's UTC date as `YYYY-MM-DD`, counted day by day from 1970-01-01 (an oracle
/// independent of the crate's date arithmetic).
fn utc_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after 1970")
        .as_secs();
    let mut days = secs / 86_400;
    let mut year = 1970;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if days < len {
            let months = [
                31,
                if leap { 29 } else { 28 },
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ];
            let mut month = 0;
            while days >= months[month] {
                days -= months[month];
                month += 1;
            }
            return format!("{year:04}-{:02}-{:02}", month + 1, days + 1);
        }
        days -= len;
        year += 1;
    }
}

/// The E-SETTINGS warning and the set-aside file: exactly one warning, carrying the code
/// and naming the new file, which holds `text`; the defaults load and `settings.toml` is
/// gone.
fn assert_set_aside(name: &str, text: &[u8]) {
    let dir = scratch(name);
    let path = dir.join("settings.toml");
    std::fs::write(&path, text).expect("write the test file");
    let before = utc_today();
    let (s, warnings) = load_settings(&path).expect("load");
    let after = utc_today();
    assert_eq!(s, Settings::default());
    let files = names(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    let kept = &files[0];
    assert!(
        *kept == format!("settings.toml.unknown-{before}")
            || *kept == format!("settings.toml.unknown-{after}"),
        "{kept}"
    );
    assert_eq!(std::fs::read(dir.join(kept)).expect("kept file"), text);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].starts_with("E-SETTINGS"), "{warnings:?}");
    assert!(warnings[0].contains(kept.as_str()), "{warnings:?}");

    // A second set-aside the same day keeps both files (Q-51).
    std::fs::write(&path, text).expect("write the test file again");
    let (_, warnings) = load_settings(&path).expect("load again");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let files = names(&dir);
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(files.iter().any(|f| f == &format!("{kept}-2")), "{files:?}");
}

/// U-29 major version 2: the file is set aside as `settings.toml.unknown-<date>` and the
/// defaults load with an E-SETTINGS warning (§1.11, §1.15).
#[test]
fn u29_major_version_2_is_set_aside_and_defaults_load() {
    assert_set_aside(
        "major_2",
        b"format = \"ftgsxr-settings\"\nversion = 2\n\n[general]\nui_scale = 1.2\n",
    );
    assert_set_aside(
        "major_2_string",
        b"format = \"ftgsxr-settings\"\nversion = \"2.0\"\n",
    );
}

/// §1.15 E-SETTINGS ("an unreadable one", I-10's corrupt file): a file that is not TOML,
/// not UTF-8, or of another format is set aside the same way (Q-51).
#[test]
fn u29_unreadable_files_are_set_aside_and_defaults_load() {
    assert_set_aside(
        "not_toml",
        b"format = \"ftgsxr-settings\"\nversion = 1\n[general\n",
    );
    assert_set_aside(
        "not_utf8",
        b"format = \"ftgsxr-settings\"\nversion = 1\n# \xff\xfe\n",
    );
    assert_set_aside("other_format", b"format = \"ftgsxr-scene\"\nversion = 1\n");
}

/// §1.11's versioning rule: a minor version is ignored, in either TOML form.
#[test]
fn u29_a_minor_version_is_ignored() {
    for version in ["\"1.7\"", "1.7", "\"1\""] {
        let (dir, s, warnings) = load_text(
            "minor",
            &format!(
                "format = \"ftgsxr-settings\"\nversion = {version}\n[general]\nui_scale = 1.2\n"
            ),
        );
        assert!(warnings.is_empty(), "{version}: {warnings:?}");
        assert_eq!(s.general.ui_scale, 1.2, "{version}");
        assert_eq!(names(&dir), ["settings.toml"], "{version}");
    }
}

/// Q-51: a file without `format` or `version` is read as version 1, with a warning
/// for each.
#[test]
fn u29_missing_format_and_version_warn_and_read_as_version_1() {
    let (_, s, warnings) = load_text("no_header", "[playback]\nloop = \"once\"\n");
    assert_eq!(s.playback.loop_mode, LoopMode::Once);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings.iter().any(|w| w.contains("`format`")),
        "{warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("`version`")),
        "{warnings:?}"
    );
}

/// U-29 atomic write: the new file is a temporary file renamed over the old one, so a
/// hard link to the old file keeps the old bytes (an in-place write would change them),
/// and no temporary file is left; a missing directory is created (Q-51).
#[test]
fn u29_the_write_is_atomic() {
    let dir = scratch("atomic");
    let path = dir.join("config").join("settings.toml");
    save_settings(&path, &Settings::default()).expect("first save");
    let old = std::fs::read(&path).expect("old bytes");
    let link = dir.join("old-link");
    std::fs::hard_link(&path, &link).expect("hard link");

    save_settings(&path, &non_default()).expect("second save");
    assert_eq!(std::fs::read(&link).expect("link bytes"), old);
    assert_ne!(std::fs::read(&path).expect("new bytes"), old);
    assert_eq!(names(&dir.join("config")), ["settings.toml"]);
    let (s, warnings) = load_settings(&path).expect("load");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s, non_default());
}

/// §2.7 Fixtures: the committed sample `settings/v1.toml` (hand-written, every key away
/// from its default) loads with no warning, as `non_default`; `fixtures_dir` checks its
/// SHA-256 against `SHA256SUMS` first.
#[test]
fn u29_the_sample_settings_fixture_loads() {
    let path = ftgsxr_testkit::fixtures_dir()
        .join("settings")
        .join("v1.toml");
    let (s, warnings) = load_settings(&path).expect("load the sample");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s, non_default());
    assert!(path.exists(), "a readable file is never set aside");
}
