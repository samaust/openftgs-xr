// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr-app`'s integration tests, one test binary (#413, §2.7 Organisation): each
//! `tests/*.rs` file is a module of it, and the support modules are declared here once.

use std::panic::Location;
use std::path::Path;

/// The stem of the `tests/*.rs` file that calls it, `u31_sources` for
/// `tests/u31_sources.rs`: through `#[track_caller]` helpers, the first caller that isn't
/// one. Each file passes it to `scratch_dir` as `binary`, so that it keeps a scratch
/// folder of its own (Q-62, #413), and names its tests by it in a child run.
#[track_caller]
fn test_file_stem() -> &'static str {
    let file = Path::new(Location::caller().file());
    assert!(
        file.parent().and_then(Path::file_name) == Some("tests".as_ref()),
        "{} is not a tests/*.rs file (a support helper without #[track_caller]?)",
        file.display()
    );
    file.file_stem()
        .and_then(|s| s.to_str())
        .expect("a test file's stem is UTF-8")
}

mod support;

mod app_plugin;
mod audio_spatial;
mod b_benchmarks;
mod dock_sections;
mod environment_mode;
mod gpu_exit_wait;
mod gpu_teardown;
mod i01_load_unload;
mod i02_http_library;
mod i03_downloads;
mod i04_pose_actions;
mod i05_audio;
mod i05_playback;
mod i06_desktop_bindings;
mod i07_mouse_selection;
mod i08_nudge_turn;
mod i09_scene;
mod i10_settings;
mod i11_panel_pointer;
mod i12_governor;
mod i13_errors;
mod i13_network;
mod i13_toasts;
mod i14_screenshot;
mod i18_audio;
mod i19_paths;
mod i19_pick;
mod i20_drop;
mod i21_net_index;
mod i23_trim;
mod keymap_names;
mod library_first_run;
mod library_refresh;
mod live_tier;
mod manipulate_content_group;
mod manipulate_grab_points;
mod manipulate_headset_fit;
mod manipulate_intents;
mod overlay_toggles;
mod panel_mode;
mod r15_render;
mod r17_governor;
mod rig_camera_controller;
mod rig_recenter_teleport;
mod scene_anchor;
mod u17_locomotion;
mod u22_governor;
mod u32_panel_hit;
mod u32_pick_instance;
mod u33_import_export;
mod u33_info;
mod u33_validate;
mod u34_headless_error_catalog;
