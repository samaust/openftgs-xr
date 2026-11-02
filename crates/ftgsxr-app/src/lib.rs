// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr application: everything the desktop and Quest builds share (§1.5–§1.7,
//! §1.10–§1.13, §2.1). M-2 adds the headless commands (#117, #118).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod headless;
pub mod persist;
pub mod platform;

// Q-67: `BenchReport` and `cmd_bench`.
pub use headless::{
    BenchReport, HeadlessError, cmd_bench, cmd_export, cmd_import, cmd_info, cmd_render,
    cmd_validate, exit_code,
};
// §2.1 `ftgsxr_app::persist` (M-3: `SettingsRes` only). Q-127
pub use persist::SettingsRes;
// §2.1 `ftgsxr_app::platform` (M-3: `Permission` only).
pub use platform::Permission;
// Q-89: the tier `cmd_render` (and M-4's `AppPlugin`) takes, for callers that depend on
// the app alone (§2.1's `ftgsxr-desktop`).
pub use ftgsxr_render::QualityTier;
// Q-129: what the Quest's provisional harness (`ftgsxr_quest::main` before M-4's
// `AppPlugin`) takes, for callers that depend on the app alone (§2.1's `ftgsxr-quest`).
pub use ftgsxr_io::{AppPaths, ModelBytes, Settings, load_settings};
pub use ftgsxr_render::{SortMode, SplatLoad, SplatPlugin, SplatTime};
// Q-T225.1: what the Quest's device test `dv2` reads of the adapter (§3.4, Dv-2).
pub use ftgsxr_render::{SplatCamera, SplatEvent, SplatStats};
// Q-135: what the Quest's Dv-3 runner (R-4 on the device, Q-122) takes to load a model,
// upload it, run stages 1–3 and read the frame back, for callers that depend on the app
// alone (§2.1's `ftgsxr-quest`).
pub use ftgsxr_core::{
    Camera, DMat3, DMat4, DVec3, Pose, TemporalIndex, UVec2, Vec3, open_model, pack_for_gpu,
    sort_key, sort_key16,
};
pub use ftgsxr_render::{
    ColorTransform, FrameBuffers, FrameContents, GpuModel, ModelDraw, Overlays, RenderMode,
    SortKey, SortPose, SplatDevice, SplatPasses, ViewParams,
};
