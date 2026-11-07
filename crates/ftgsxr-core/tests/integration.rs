// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr-core`'s integration tests, one test binary (#413, §2.7 Organisation): each
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

mod ds_kitchen_boxes;
mod types_api;
mod u01_read_exchange;
mod u01_render_conventions;
mod u02_model_validate;
mod u02_reader_rejections;
mod u03_safetensors_container;
mod u04_model_identity;
mod u05_checkpoint_activation;
mod u06_ply_snapshot;
mod u07_motion;
mod u07_rebased_time;
mod u08_index;
mod u08_motion_window;
mod u09_sh;
mod u10_covariance;
mod u11_camera_from_stage_pose;
mod u11_project;
mod u12_alpha_at;
mod u13_sort_key;
mod u14_clock;
mod u14_model_time;
mod u15_ray_hit;
mod u16_grab_transforms;
mod u18_presets_fit_floor;
mod u19_occlusion_weight;
mod u23_layout;
mod u24_render_request;
mod u25_renders;
mod u26_info_json;
mod u34_core_error_catalog;
mod u35_skeleton;
mod u36_scene_manifest;
mod u38_model_bounds;
mod u39_trim;
mod write_exchange;
