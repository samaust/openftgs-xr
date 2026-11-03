// SPDX-License-Identifier: Apache-2.0

//! Settings, scenes and poses to and from the world (§2.1 `ftgsxr_app::persist`). M-3
//! brings `SettingsRes`, which `ftgsxr_xr::session` reads (Q-127); #277 adds
//! `SettingsPath` and the settings' load at `AppPlugin`'s build (Q-163); the commands,
//! the saves and the scene functions are #285's.

use std::path::PathBuf;

use bevy::prelude::{Resource, World};
use ftgsxr_io::{Settings, load_settings};

/// The player's settings in the world (§1.11, §2.1).
#[derive(Resource, Clone, Debug, Default)]
pub struct SettingsRes(pub Settings);

/// The settings file the player reads at startup and saves to (§1.7 `--settings FILE`,
/// §1.11): inserted by the caller to override the platform's `settings.toml`, else by
/// `AppPlugin` with that default.
// Q-163
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct SettingsPath(pub PathBuf);

/// Q-163: when `SettingsRes` is absent, reads the settings from `SettingsPath`, else
/// from `default` (inserting `SettingsPath` with it), and inserts `SettingsRes`; each
/// warning is logged. A `SettingsRes` the caller inserted is left alone.
pub(crate) fn load_into(world: &mut World, default: PathBuf) {
    let path = match world.get_resource::<SettingsPath>() {
        Some(p) => p.0.clone(),
        None => {
            world.insert_resource(SettingsPath(default.clone()));
            default
        }
    };
    if world.contains_resource::<SettingsRes>() {
        return;
    }
    let settings = match load_settings(&path) {
        Ok((settings, warnings)) => {
            for w in warnings {
                bevy::log::warn!("{}: {w}", path.display());
            }
            settings
        }
        Err(e) => {
            // E-SETTINGS and E-IO are recoverable (§1.15): the defaults, with the warning.
            bevy::log::warn!("{}: {}: {e}", path.display(), e.code());
            Settings::default()
        }
    };
    world.insert_resource(SettingsRes(settings));
}
