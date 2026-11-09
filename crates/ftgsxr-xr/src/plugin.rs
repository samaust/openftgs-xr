// SPDX-License-Identifier: Apache-2.0

//! `XrPlugin` and its composition with Bevy's plugins (§2.1, Q-120).

use bevy::app::{App, Plugin, PluginGroup, PluginGroupBuilder, PostUpdate};
use bevy::log::info;
use bevy::prelude::IntoScheduleConfigs;
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

use crate::anchors::AnchorsPlugin;
use crate::depth::DepthPlugin;
use crate::display::DisplayPlugin;
use crate::error::{XrError, log_runtime, log_warning};
use crate::light::LightPlugin;
use crate::panel::PanelPlugin;
use crate::passthrough::{self, Environment, EnvironmentMode, PassthroughStyle};
use crate::session::{SessionConfig, SessionPlugin};
use crate::tracking::{self, FREQUENCY_HINT_EXT, TrackingPlugin};
use crate::xr_input::XrInputPlugin;

/// The XR build's plugin (§2.1, §1.8): bevy\_mod\_openxr's session, init, events,
/// render, camera and action plugins with the player's app info, Appendix E.1's
/// extensions where the runtime offers them, an OPAQUE (D-14) `Rgba8UnormSrgb` (Q-91)
/// session, D-24's reference spaces and the eyes as `SplatCamera`s. Add it through
/// [`add_xr`] (Q-120). `hands` asks for §1.8's
/// hand extensions (PRD #452, #454); `sort` is each eye's `SplatCamera.sort` (Q-121).
/// `passthrough` builds `passthrough`'s layer, switch and style (§1.9, D-14), which run
/// where the runtime offers XR\_FB\_passthrough; without it the mode stays VR.
#[derive(Clone, Copy, Debug)]
pub struct XrPlugin {
    /// Whether passthrough is wanted: true on the Quest and for `play --xr` (§1.9).
    pub passthrough: bool,
    /// Whether hand tracking is wanted: true on the Quest and for `play --xr` (PRD #452).
    pub hands: bool,
    /// What each eye's `SplatCamera` sorts with (Q-121).
    pub sort: SortMode,
}

/// The extensions of Appendix E.1 the milestone uses, with §1.8's hand extensions when
/// `hands` is set; the crate drops those the runtime lacks with a warning, and adds
/// XR\_KHR\_vulkan\_enable2 itself.
fn extensions(hands: bool) -> OxrExtensions {
    let mut exts = OxrExtensions::default();
    let raw = exts.raw_mut();
    raw.khr_vulkan_enable2 = true;
    raw.ext_local_floor = true;
    // §1.9: `passthrough`'s layer, and `Environment.passthrough_available`.
    raw.fb_passthrough = true;
    raw.fb_display_refresh_rate = true;
    raw.fb_foveation = true;
    raw.fb_foveation_configuration = true;
    raw.fb_foveation_vulkan = true;
    raw.fb_swapchain_update_state = true;
    raw.fb_swapchain_update_state_vulkan = true;
    raw.meta_recommended_layer_resolution = true;
    // §1.9's anchors (D-26, #571): the entity, its query by UUID and its persistence.
    raw.fb_spatial_entity = true;
    raw.fb_spatial_entity_query = true;
    raw.meta_spatial_entity_persistence = true;
    // D-26: spike check 5's fallback, the discovery by UUID.
    raw.meta_spatial_entity_discovery = true;
    // §1.9's depth occlusion (E13, #570).
    raw.meta_environment_depth = true;
    // The Touch Plus profile (Appendix D.2) through `tracking`'s bindings.
    raw.meta_touch_controller_plus = true;
    // §1.8 Hands (PRD #452, #454); the frequency hint by name, since no pinned openxr
    // release carries it (Q-275).
    raw.ext_hand_tracking = hands;
    raw.fb_hand_tracking_aim = hands;
    raw.meta_simultaneous_hands_and_controllers = hands;
    // D-55: the runtime's in-hand detection, so a set-down controller leaves its side
    // to the hand (#540).
    raw.meta_detached_controllers = hands;
    if hands {
        raw.other.push(FREQUENCY_HINT_EXT.to_vec());
    }
    exts
}

/// The hand extensions of §1.8 absent from `enabled`, in Appendix E.1's order (the
/// frequency hint by name, Q-275), then XR\_META\_detached\_controllers (D-55).
// D-55
fn missing_hand_extensions(enabled: &openxr::ExtensionSet) -> Vec<&'static str> {
    [
        ("XR_EXT_hand_tracking", enabled.ext_hand_tracking),
        ("XR_FB_hand_tracking_aim", enabled.fb_hand_tracking_aim),
        (
            "XR_META_hand_tracking_frequency_hint",
            enabled
                .other
                .iter()
                .any(|n| n.as_slice() == FREQUENCY_HINT_EXT),
        ),
        (
            "XR_META_simultaneous_hands_and_controllers",
            enabled.meta_simultaneous_hands_and_controllers,
        ),
        (
            "XR_META_detached_controllers",
            enabled.meta_detached_controllers,
        ),
    ]
    .into_iter()
    .filter_map(|(name, on)| (!on).then_some(name))
    .collect()
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
            "XrPlugin: passthrough {} and hands {} wanted, sort {:?}",
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
                exts: extensions(self.hands),
                ..OxrInitPlugin::default()
            });
        if app.world().get_resource::<XrState>() == Some(&XrState::Unavailable) {
            // E-XR-RUNTIME: the desktop continues without XR, the Quest exits 2 (§1.15);
            // `play --xr` exits 2 through `runtime_unavailable` (Q-162).
            log_runtime(NO_INSTANCE);
        }
        // E-XR-UNSUPPORTED once per absent hand extension, as a warning with no toast
        // (Q-134; Q-161: hands are not an explicit request). Without an instance there is
        // no extension set, and E-XR-RUNTIME above says why.
        if self.hands
            && let Some(enabled) = app.world().get_resource::<OxrEnabledExtensions>()
        {
            for name in missing_hand_extensions(enabled.0.raw()) {
                log_warning(&XrError::Unsupported(name));
            }
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
/// (`tracking` locates the hands itself, PRD #452; the overlay is not the player's).
struct XrTail {
    sort: SortMode,
    hands: bool,
    passthrough: bool,
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
        .add_plugins(TrackingPlugin { hands: self.hands })
        // Appendix D.2 and D.3 to intents (#460), after `tracking`'s `XrHands`.
        .add_plugins(XrInputPlugin)
        // `passthrough`'s mode check of the hand and controller meshes, before `tracking`
        // places them (§2.1's module order: `tracking` cannot read `Environment`).
        .add_systems(
            PostUpdate,
            passthrough::devices_drawn.before(tracking::DrawDevices),
        )
        // The scene anchor's handshake and the store (#571), after `session`, whose
        // twist it uses, and `tracking`, whose permission flow it shares.
        .add_plugins(AnchorsPlugin)
        // The panel and the wrist menu (#462), after `xr_input` (§2.1's module order).
        .add_plugins(PanelPlugin)
        // After the events plugin and `session`, whose re-creation it uses (Q-124).
        .add_plugins(DisplayPlugin)
        // §1.9 (#569, Q-342): the contact shadows, drawn in passthrough mode only, where
        // Bevy's `PbrPlugin` is in the group.
        .add_plugins(passthrough::ContactShadowPlugin);
        // §1.9, D-14 (#568): the layer, the switch and the style; without it nothing is
        // created and the mode stays VR (M-3's behaviour).
        if self.passthrough {
            app.add_plugins(passthrough::PassthroughPlugin);
            // §1.9 Depth occlusion (#570): the Depth API needs passthrough (E.2).
            app.add_plugins(DepthPlugin);
            // §1.9 Light estimation (#668): the camera runs in passthrough mode only (Q-385).
            app.add_plugins(LightPlugin);
        }
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
        .add(XrTail {
            sort: xr.sort,
            hands: xr.hands,
            passthrough: xr.passthrough,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §1.8 Hands, PRD #452's "What `XrPlugin.hands` means": the hand extensions, with
    /// D-55's XR\_META\_detached\_controllers, are asked for only when `hands` is set.
    #[test]
    fn extensions_ask_for_the_hand_extensions_only_with_hands() {
        let with = extensions(true);
        let with = with.raw();
        assert!(with.ext_hand_tracking);
        assert!(with.fb_hand_tracking_aim);
        assert!(with.meta_simultaneous_hands_and_controllers);
        // D-55: the held-controller signal.
        assert!(with.meta_detached_controllers);
        // Q-275: the frequency hint by name.
        assert_eq!(
            with.other,
            vec![b"XR_META_hand_tracking_frequency_hint\0".to_vec()]
        );
        let without = extensions(false);
        let without = without.raw();
        assert!(!without.ext_hand_tracking);
        assert!(!without.fb_hand_tracking_aim);
        assert!(!without.meta_simultaneous_hands_and_controllers);
        assert!(!without.meta_detached_controllers);
        assert!(without.other.is_empty());
        // The rest of the set is the same either way.
        assert!(with.fb_passthrough && without.fb_passthrough);
    }

    /// The absent hand extensions, in Appendix E.1's order, each named once.
    #[test]
    fn missing_hand_extensions_names_each_absent_one() {
        let mut enabled = OxrExtensions::default();
        assert_eq!(
            missing_hand_extensions(enabled.raw()),
            vec![
                "XR_EXT_hand_tracking",
                "XR_FB_hand_tracking_aim",
                "XR_META_hand_tracking_frequency_hint",
                "XR_META_simultaneous_hands_and_controllers",
                "XR_META_detached_controllers",
            ]
        );
        enabled
            .raw_mut()
            .other
            .push(b"XR_META_hand_tracking_frequency_hint\0".to_vec());
        enabled.raw_mut().ext_hand_tracking = true;
        enabled.raw_mut().meta_simultaneous_hands_and_controllers = true;
        enabled.raw_mut().meta_detached_controllers = true;
        assert_eq!(
            missing_hand_extensions(enabled.raw()),
            vec!["XR_FB_hand_tracking_aim"]
        );
        enabled.raw_mut().fb_hand_tracking_aim = true;
        assert!(missing_hand_extensions(enabled.raw()).is_empty());
    }
}
