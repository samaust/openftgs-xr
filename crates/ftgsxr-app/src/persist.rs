// SPDX-License-Identifier: Apache-2.0

//! Settings, scenes and poses to and from the world (§2.1 `ftgsxr_app::persist`). M-3
//! brings `SettingsRes` only, which `ftgsxr_xr::session` reads (Q-T223.1); the commands
//! and the scene functions are M-4's.

use bevy::prelude::Resource;
use ftgsxr_io::Settings;

/// The player's settings in the world (§1.11, §2.1).
#[derive(Resource, Clone, Debug, Default)]
pub struct SettingsRes(pub Settings);
