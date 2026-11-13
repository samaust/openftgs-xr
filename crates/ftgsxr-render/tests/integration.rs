// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr-render`'s integration tests, one test binary (#413, §2.7 Organisation): each
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

mod cull_support;
mod display_support;
mod probe_support;

mod bevy_adapter;
mod bevy_adapter_m4;
mod bevy_adapter_m6;
mod bevy_adapter_m8;
mod concurrent_compile;
mod gpu_foundation;
mod import_vulkan_texture;
mod load_splat_asset;
mod occlusion_edges;
mod quality_tiers;
mod r01_probes;
mod r02_additional_probes;
mod r03_trained_models;
mod r04_sort;
mod r05_cull_against_project;
mod r06_count_and_overflow;
mod r07_several_models;
mod r08_pose;
mod r09_shared_center_stereo;
mod r10_tier_bounds;
mod r11_color_space;
mod r12_occlusion;
mod r13_centers_mode;
mod r14_light_transform;
mod r19_transmittance_bound;
mod r20_trim_cull;
mod r21_layer_probe;
mod sh_ranges;
mod shared_center_records;
mod stereo_stats;
mod u20_manual_transform;
mod u34_render_error_catalog;
mod warm_up_teardown;
