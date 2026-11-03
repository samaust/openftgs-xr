// SPDX-License-Identifier: Apache-2.0

//! `XrPlugin` and its composition with Bevy's plugins (§2.1, Q-120).

use bevy::app::{App, Plugin, PluginGroup, PluginGroupBuilder};
use bevy::log::info;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy_mod_openxr::action_binding::OxrActionBindingPlugin;
use bevy_mod_openxr::action_set_attaching::OxrActionAttachingPlugin;
use bevy_mod_openxr::action_set_syncing::OxrActionSyncingPlugin;
use bevy_mod_openxr::exts::{OxrEnabledExtensions, OxrExtensions};
use bevy_mod_openxr::init::OxrInitPlugin;
use bevy_mod_openxr::poll_events::OxrEventsPlugin;
use bevy_mod_openxr::render::OxrRenderPlugin;
use bevy_mod_openxr::resources::OxrSessionConfig;
use bevy_mod_openxr::spaces::{OxrSpacePatchingPlugin, OxrSpatialPlugin};
use bevy_mod_openxr::types::{AppInfo, Version};
use bevy_mod_xr::camera::XrCameraPlugin;
use bevy_mod_xr::session::{XrSessionPlugin, XrState};
use ftgsxr_render::SortMode;
use openxr::EnvironmentBlendMode;

use crate::display::DisplayPlugin;
use crate::error::{XrError, log_runtime};
use crate::passthrough::{Environment, EnvironmentMode, PassthroughStyle};
use crate::session::{SessionConfig, SessionPlugin};
use crate::tracking::TrackingPlugin;

/// The XR build's plugin (§2.1, §1.8): bevy\_mod\_openxr's session, init, events,
/// render, camera and action plugins with the player's app info, Appendix E.1's
/// extensions where the runtime offers them, an OPAQUE (D-14) `Rgba8UnormSrgb` (Q-91)
/// session, D-24's reference spaces and the eyes as `SplatCamera`s. Add it through
/// [`add_xr`] (Q-120). `passthrough` and `hands` are carried and logged in M-3; `sort` is
/// each eye's `SplatCamera.sort` (Q-121).
#[derive(Clone, Copy, Debug)]
pub struct XrPlugin {
    /// Whether passthrough is wanted (M-6).
    pub passthrough: bool,
    /// Whether hand tracking is wanted (M-5).
    pub hands: bool,
    /// What each eye's `SplatCamera` sorts with (Q-121).
    pub sort: SortMode,
}

/// The extensions of Appendix E.1 the milestone uses; the crate drops those the runtime
/// lacks with a warning, and adds XR\_KHR\_vulkan\_enable2 itself.
fn extensions() -> OxrExtensions {
    let mut exts = OxrExtensions::default();
    let raw = exts.raw_mut();
    raw.khr_vulkan_enable2 = true;
    raw.ext_local_floor = true;
    // For `Environment.passthrough_available` only in M-3.
    raw.fb_passthrough = true;
    raw.fb_display_refresh_rate = true;
    raw.fb_foveation = true;
    raw.fb_foveation_configuration = true;
    raw.fb_foveation_vulkan = true;
    raw.fb_swapchain_update_state = true;
    raw.fb_swapchain_update_state_vulkan = true;
    raw.meta_recommended_layer_resolution = true;
    // The Touch Plus profile (Appendix D.2) through `tracking`'s bindings.
    raw.meta_touch_controller_plus = true;
    exts
}

/// The player's name and version for the OpenXR instance.
fn app_info() -> AppInfo {
    let mut v = env!("CARGO_PKG_VERSION")
        .split('.')
        .map(|p| p.parse::<u16>().unwrap_or(0));
    let (major, minor, patch) = (
        v.next().unwrap_or(0),
        v.next().unwrap_or(0),
        v.next().unwrap_or(0),
    );
    AppInfo {
        name: "openftgs-xr".into(),
        version: Version(
            u8::try_from(major).unwrap_or(u8::MAX),
            u8::try_from(minor).unwrap_or(u8::MAX),
            patch,
        ),
    }
}

impl Plugin for XrPlugin {
    fn build(&self, app: &mut App) {
        info!(
            "XrPlugin: passthrough {} and hands {} wanted (M-6, M-5), sort {:?}",
            self.passthrough, self.hands, self.sort
        );
        app.add_plugins(XrSessionPlugin { auto_handle: true })
            .insert_resource(OxrSessionConfig {
                // D-14
                blend_mode_preference: vec![EnvironmentBlendMode::OPAQUE],
                // Q-91
                formats: Some(vec![TextureFormat::Rgba8UnormSrgb]),
                resolutions: None,
            });
        // Q-139: the Quest's runtime asks for VK_KHR_surface_protected_capabilities
        // without the instance extension it requires (a validation error at instance
        // creation, Dv-2), so the instance also enables VK_KHR_get_surface_capabilities2.
        #[cfg(target_os = "android")]
        app.insert_resource(bevy_mod_openxr::graphics::OxrManualGraphicsConfig {
            fallback_backend: bevy_mod_openxr::graphics::GraphicsBackend::Vulkan(()),
            vk_instance_exts: vec![c"VK_KHR_get_surface_capabilities2"],
            vk_device_exts: Vec::new(),
        });
        app
            // The crate creates the instance here and adds `RenderPlugin` itself, with
            // Bevy's default render plugin when the instance fails.
            .add_plugins(OxrInitPlugin {
                app_info: app_info(),
                exts: extensions(),
                ..OxrInitPlugin::default()
            });
        if app.world().get_resource::<XrState>() == Some(&XrState::Unavailable) {
            // E-XR-RUNTIME: the desktop continues without XR, the Quest exits 2 (§1.15);
            // `play --xr` exits 2 through `runtime_unavailable` (Q-162).
            log_runtime(NO_INSTANCE);
        }
        let passthrough_available = app
            .world()
            .get_resource::<OxrEnabledExtensions>()
            .is_some_and(|e| e.fb_passthrough);
        app.insert_resource(Environment {
            mode: EnvironmentMode::Vr,
            passthrough_available,
            style: PassthroughStyle::default(),
        });
    }
}

/// What E-XR-RUNTIME says when the loader finds no runtime.
const NO_INSTANCE: &str = "the OpenXR instance could not be created";

/// E-XR-RUNTIME when `app`, built with [`add_xr`], has no OpenXR instance: the loader
/// found no runtime, or the runtime refused the instance (Q-162: `play --xr` then exits
/// 2). `None` once an instance exists, whatever happens to its session later (§1.15's
/// recovery), and in an app without `XrPlugin`.
// Q-162, Q-220
pub fn runtime_unavailable(app: &App) -> Option<XrError> {
    (app.world().get_resource::<XrState>() == Some(&XrState::Unavailable))
        .then(|| XrError::Runtime(NO_INSTANCE.to_owned()))
}

/// The plugins that follow Bevy's group (after `PipelinedRenderingPlugin`, which
/// `OxrRenderPlugin` looks for). The hand-tracking and overlay plugins are left out
/// (hands are M-5's; the overlay is not the player's).
struct XrTail {
    sort: SortMode,
}

impl Plugin for XrTail {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            OxrEventsPlugin,
            SessionPlugin {
                config: SessionConfig { sort: self.sort },
            },
            OxrRenderPlugin::default(),
            XrCameraPlugin,
            OxrActionAttachingPlugin,
            OxrActionBindingPlugin,
            OxrActionSyncingPlugin,
            OxrSpatialPlugin,
            OxrSpacePatchingPlugin,
        ))
        // After the binding plugin, whose schedule `tracking` adds to.
        .add_plugins(TrackingPlugin)
        // After the events plugin and `session`, whose re-creation it uses (Q-124).
        .add_plugins(DisplayPlugin);
    }
}

/// Composes `plugins` with `xr` (Q-120): disables `RenderPlugin` in the group, inserts
/// `xr` before it (its init adds `RenderPlugin` back) and the rest of the XR plugins at
/// the end. Works with or without winit.
pub fn add_xr(plugins: impl PluginGroup, xr: XrPlugin) -> PluginGroupBuilder {
    plugins
        .build()
        .disable::<RenderPlugin>()
        .add_before::<RenderPlugin>(xr)
        .add(XrTail { sort: xr.sort })
}
