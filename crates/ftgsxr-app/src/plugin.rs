// SPDX-License-Identifier: Apache-2.0

//! `AppPlugin` (§2.1 `ftgsxr_app::plugin`): what both builds add to their `App`, the
//! splat adapter, the platform, the settings and the feature modules' plugins.

use std::path::PathBuf;
use std::sync::Arc;

use bevy::app::{App, Plugin};
use bevy::render::RenderApp;
use bevy_egui::EguiPlugin;
use ftgsxr_io::ModelRef;
use ftgsxr_render::{QualityTier, SplatPlugin};

use crate::governor::GovernorPlugin;
use crate::library::LibraryPlugin;
use crate::persist;
use crate::platform::{Platform, PlatformRes};
use crate::playback::PlaybackPlugin;

pub use crate::desktop_log::init_desktop_log;

/// The application (§2.1): adds `SplatPlugin { tier }` (PRD #275), inserts `PlatformRes`,
/// loads the settings into `SettingsRes` when the resource is absent (Q-163), and adds
/// the feature modules' plugins. Bevy's `LogPlugin` is left to the caller to disable:
/// each build installs its own logger (Q-151).
pub struct AppPlugin {
    /// The platform port.
    pub platform: Arc<dyn Platform>,
    /// The quality tier (§1.12).
    pub tier: QualityTier,
    /// The models to load at startup (§1.7 `play`).
    pub initial: Vec<ModelRef>,
    /// The scene file to load at startup (§1.7 `play`).
    pub scene: Option<PathBuf>,
}

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SplatPlugin {
            tier: self.tier.clone(),
        })
        .insert_resource(PlatformRes(self.platform.clone()));
        // Q-163
        let default = self.platform.paths().config.join("settings.toml");
        persist::load_into(app.world_mut(), default);
        // PRD #275's spike check 1 failed (bevy_egui 0.42's build loads its shader into
        // `Assets<Shader>`, absent without `RenderPlugin`); its fallback: egui is added
        // only when the render app exists, so `RenderPlugin` comes before `AppPlugin`.
        if app.get_sub_app(RenderApp).is_some() && !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        // `initial` loads through the library; `scene` is `persist`'s (#285).
        app.add_plugins(LibraryPlugin {
            initial: self.initial.clone(),
        });
        app.add_plugins(PlaybackPlugin);
        // E16 (#288).
        app.add_plugins(GovernorPlugin);
    }
}
