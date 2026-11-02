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
