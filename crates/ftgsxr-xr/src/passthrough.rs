// SPDX-License-Identifier: Apache-2.0

//! The passthrough layer, style and mode switching (§1.9, D-14, §2.1 `Environment`).
//!
//! - The switch (main world, Q-335): `[environment] mode` is the wanted mode. On the
//!   headset (the `Headset` marker) `Toggle(Feature::Passthrough)` flips it; while a
//!   session exists `Environment.mode` is the effective mode, passthrough when the
//!   setting asks, the marker exists and the runtime offers XR\_FB\_passthrough; a setting
//!   asking for passthrough on a headset whose runtime lacks it is written back to `vr`
//!   with E-XR-UNSUPPORTED's toast once per session (Q-161). Each eye's
//!   `SplatCamera.output` follows the mode: no background in passthrough, `[environment]
//!   background` in VR mode (Q-94, Q-101, Q-234).
//! - The style (Q-337): the three `passthrough_*` settings, mapped to the extension's
//!   brightness, contrast and saturation and sent at every change.
//! - The layer (render world, `layer`): created at session creation, submitted first in
//!   bevy\_mod\_openxr's layer list and resumed in passthrough mode, paused and removed in
//!   VR mode; the session stays OPAQUE (D-14).
//! - The contact shadows (`shadow`, Q-342): `ContactShadowPlugin`, an ellipse on the floor
//!   under each visible instance in passthrough mode on the headset.
//! - `XrFeatures.passthrough` (Q-336): `Available` on the headset while a session exists
//!   and the runtime offers the extension, else `Absent`.
//!
//! Built only with `XrPlugin.passthrough`; without it nothing is created and the mode
//! stays VR (M-3's behaviour).

// Dv-6's device side (Q-346, Q-308), built for the device and the host tests.
#[cfg(any(test, target_os = "android"))]
mod dv6;
mod layer;
// The contact shadows (Q-342, #569).
mod shadow;

pub use shadow::ContactShadowPlugin;

use bevy::app::{App, Plugin, PostUpdate};
use bevy::ecs::message::MessageReader;
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::log::info;
use bevy::prelude::{
    Commands, DetectChangesMut, IntoScheduleConfigs, Query, Res, ResMut, Resource, With, not,
    resource_exists,
};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrSessionCreated;
use ftgsxr_app::{
    Feature, FeatureState, Headset, Intent, SettingsRes, Toast, ToastLevel, XrFeatures,
};
use ftgsxr_io::Settings;
use ftgsxr_render::{OutputSpace, SplatCamera};

use crate::XrError;
use crate::error::log_warning;
use crate::tracking::DevicesDrawn;

/// The extension the passthrough layer needs (Appendix E.1).
const EXTENSION: &str = "XR_FB_passthrough";

/// The environment mode (§1.9): VR, the projection layer alone and opaque, or passthrough.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EnvironmentMode {
    /// The projection layer alone, opaque, over §1.4's background.
    #[default]
    Vr,
    /// A passthrough layer under the projection layer (D-14).
    Passthrough,
}

/// The passthrough's appearance (§1.9), each control in \[−1, 1\].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PassthroughStyle {
    /// Brightness, in \[−1, 1\].
    pub brightness: f32,
    /// Contrast, in \[−1, 1\].
    pub contrast: f32,
    /// Saturation, in \[−1, 1\].
    pub saturation: f32,
}

/// The environment's state (§1.9, §2.1): the mode, whether the runtime offers
/// passthrough (XR\_FB\_passthrough enabled on the instance; false on the desktop), and
/// the style.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    /// The current mode; `Vr` in M-3.
    pub mode: EnvironmentMode,
    /// Whether XR\_FB\_passthrough was enabled on the instance.
    pub passthrough_available: bool,
    /// The passthrough style.
    pub style: PassthroughStyle,
}

/// The extension's brightness, contrast and saturation for `style` (Q-337): b → 100·b,
/// c → 1 + c, s → 1 + s, each control clamped to §1.9's \[−1, 1\] first, so the result
/// lies in the registry's ranges (\[−100, 100\], \[0, ∞), \[0, ∞)) with 0 at their
/// neutral values (0, 1, 1).
// Q-337
pub(crate) fn style_map(style: PassthroughStyle) -> (f32, f32, f32) {
    let c = |v: f32| v.clamp(-1.0, 1.0);
    (
        100.0 * c(style.brightness),
        1.0 + c(style.contrast),
        1.0 + c(style.saturation),
    )
}

/// What the render world's layer follows: the effective mode and the style, extracted
/// each frame (§1.9; the layer list is the render world's).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, ExtractResource)]
pub(crate) struct LayerRequest {
    pub(crate) mode: EnvironmentMode,
    pub(crate) style: PassthroughStyle,
}

/// Whether E-XR-UNSUPPORTED's toast for the extension was raised in this session
/// (Q-161: once per session).
// Q-161
#[derive(Resource, Default)]
struct Toasted(bool);

/// `passthrough`'s systems, added by `plugin` when `XrPlugin.passthrough` is set.
pub(crate) struct PassthroughPlugin;

impl Plugin for PassthroughPlugin {
    fn build(&self, app: &mut App) {
        init_switch(app);
        app.add_plugins(ExtractResourcePlugin::<LayerRequest>::default())
            .add_systems(XrSessionCreated, new_session)
            .add_systems(
                PostUpdate,
                (
                    switch_systems().run_if(resource_exists::<OxrSession>),
                    no_session.run_if(not(resource_exists::<OxrSession>)),
                ),
            );
        layer::add(app);
        // Dv-6: inert until the quest runner's entity names it (Q-308, Q-346).
        #[cfg(target_os = "android")]
        dv6::add(app);
    }
}

/// The switch's resources and messages.
fn init_switch(app: &mut App) {
    app.init_resource::<LayerRequest>()
        .init_resource::<Toasted>()
        .add_message::<Intent>();
}

/// The switch, in order: the toggle, the effective mode and the style, the eyes.
fn switch_systems() -> ScheduleConfigs<ScheduleSystem> {
    (toggle_mode, switch_mode, eyes_follow_mode).chain()
}

/// A new session may toast the absent extension once more (Q-161).
fn new_session(mut toasted: ResMut<Toasted>) {
    toasted.0 = false;
}

/// Q-335: `Toggle(Feature::Passthrough)` flips `[environment] mode` on the headset; off
/// it, the toggle has no consumer.
// Q-335
fn toggle_mode(
    mut intents: MessageReader<Intent>,
    headset: Option<Res<Headset>>,
    settings: Option<ResMut<SettingsRes>>,
) {
    let toggles = intents
        .read()
        .filter(|i| **i == Intent::Toggle(Feature::Passthrough))
        .count();
    if toggles % 2 == 0 || headset.is_none() {
        return;
    }
    let Some(mut settings) = settings else {
        return;
    };
    let e = &mut settings.0.environment;
    e.mode = if e.mode == PASSTHROUGH {
        VR
    } else {
        PASSTHROUGH
    }
    .to_owned();
    info!("environment mode {} asked for", e.mode);
}

/// `[environment] mode`'s values (§1.11).
const PASSTHROUGH: &str = "passthrough";
const VR: &str = "vr";

/// Q-335: the effective mode on `Environment` and the request (passthrough when the
/// setting asks, the `Headset` marker exists and the runtime offers the extension); a
/// setting asking for passthrough on a headset without the extension written back to
/// `vr` with E-XR-UNSUPPORTED (a warning each time, the toast once per session, Q-161);
/// the style from the settings (Q-337); `XrFeatures.passthrough` (Q-336).
// Q-161, Q-335, Q-336
#[allow(clippy::too_many_arguments)]
fn switch_mode(
    headset: Option<Res<Headset>>,
    settings: Option<ResMut<SettingsRes>>,
    mut env: ResMut<Environment>,
    mut request: ResMut<LayerRequest>,
    mut toasted: ResMut<Toasted>,
    features: Option<ResMut<XrFeatures>>,
    mut commands: Commands,
) {
    let headset = headset.is_some();
    let available = env.passthrough_available;
    let default = Settings::default();
    let mut settings = settings;
    let wanted = settings
        .as_deref()
        .map_or(&default, |s| &s.0)
        .environment
        .mode
        == PASSTHROUGH;
    if wanted && headset && !available {
        if let Some(s) = settings.as_deref_mut() {
            s.0.environment.mode = VR.to_owned();
        }
        let e = XrError::Unsupported(EXTENSION);
        log_warning(&e);
        if !toasted.0 {
            toasted.0 = true;
            commands.trigger(Toast {
                text: format!("{}: {}", e.code(), e.user_text()),
                level: ToastLevel::Warning,
            });
        }
    }
    let mode = if wanted && headset && available {
        EnvironmentMode::Passthrough
    } else {
        EnvironmentMode::Vr
    };
    let e = &settings.as_deref().map_or(&default, |s| &s.0).environment;
    let style = PassthroughStyle {
        brightness: e.passthrough_brightness,
        contrast: e.passthrough_contrast,
        saturation: e.passthrough_saturation,
    };
    if env.mode != mode || env.style != style {
        env.mode = mode;
        env.style = style;
    }
    request.set_if_neq(LayerRequest { mode, style });
    // §1.9: on the desktop the tab shows the VR-mode controls only.
    let state = if headset && available {
        FeatureState::Available
    } else {
        FeatureState::Absent
    };
    if let Some(mut f) = features
        && f.passthrough != state
    {
        f.passthrough = state;
    }
}

/// Without a session there is no layer: VR mode, and the feature `Absent` (Q-336, as
/// occlusion's).
// Q-336
fn no_session(
    mut env: ResMut<Environment>,
    mut request: ResMut<LayerRequest>,
    features: Option<ResMut<XrFeatures>>,
) {
    if let Some(mut f) = features
        && f.passthrough != FeatureState::Absent
    {
        f.passthrough = FeatureState::Absent;
    }
    if env.mode != EnvironmentMode::Vr {
        env.mode = EnvironmentMode::Vr;
    }
    if request.mode != EnvironmentMode::Vr {
        request.mode = EnvironmentMode::Vr;
    }
}

/// Each eye's output follows the mode (Q-94, Q-101, Q-234): no background in
/// passthrough, `[environment] background` in VR mode. Only the form is changed, so an
/// eye made later (over the settings' background, Q-127) is caught the next frame, and
/// `rig`'s following of the background and a replayed request's override stay theirs.
// Q-94, Q-234
fn eyes_follow_mode(
    env: Res<Environment>,
    settings: Option<Res<SettingsRes>>,
    mut eyes: Query<&mut SplatCamera, With<XrCamera>>,
) {
    let background = settings.as_deref().map_or_else(
        || Settings::default().environment.background,
        |s| s.0.environment.background,
    );
    for mut eye in &mut eyes {
        let OutputSpace::LinearOver { background: now } = eye.output else {
            continue;
        };
        let want = match env.mode {
            EnvironmentMode::Passthrough => None,
            EnvironmentMode::Vr => Some(now.unwrap_or(background)),
        };
        if now != want {
            eye.output = OutputSpace::LinearOver { background: want };
        }
    }
}

/// The mode check of `tracking`'s hand and controller meshes (§1.8 Controllers and
/// Hands: drawn in VR mode, not in passthrough mode): `DevicesDrawn` follows
/// `Environment.mode`, false without an `Environment`.
pub(crate) fn devices_drawn(env: Option<Res<Environment>>, mut drawn: ResMut<DevicesDrawn>) {
    let want = DevicesDrawn(env.is_some_and(|e| e.mode == EnvironmentMode::Vr));
    if *drawn != want {
        *drawn = want;
    }
}

#[cfg(test)]
mod tests {
    use bevy::app::{App, Update};
    use bevy::prelude::On;
    use ftgsxr_render::SortMode;

    use super::*;

    /// Q-337: 0 is the extension's neutral (0, 1, 1); ±1 reach its range's ends of
    /// brightness, ±100, and contrast and saturation 2 and 0 (registry page,
    /// XrPassthroughBrightnessContrastSaturationFB).
    #[test]
    fn style_map_takes_the_three_fixed_points() {
        let at = |v: f32| {
            style_map(PassthroughStyle {
                brightness: v,
                contrast: v,
                saturation: v,
            })
        };
        assert_eq!(at(0.0), (0.0, 1.0, 1.0));
        assert_eq!(at(1.0), (100.0, 2.0, 2.0));
        assert_eq!(at(-1.0), (-100.0, 0.0, 0.0));
    }

    /// A headless app with the switch's systems in `Update` (the session's run condition
    /// left out: the systems run as while a session exists), the eyes and the toasts
    /// recorded.
    fn switch_app(available: bool, headset: bool, mode: &str) -> App {
        let mut app = App::new();
        let mut settings = Settings::default();
        settings.environment.mode = mode.to_owned();
        settings.environment.background = [0.2, 0.3, 0.4];
        app.insert_resource(SettingsRes(settings))
            .insert_resource(Environment {
                mode: EnvironmentMode::Vr,
                passthrough_available: available,
                style: PassthroughStyle::default(),
            })
            .init_resource::<XrFeatures>()
            .init_resource::<Toasts>()
            .add_message::<Intent>()
            .add_observer(|t: On<Toast>, mut toasts: ResMut<Toasts>| {
                toasts.0.push(t.event().clone());
            });
        init_switch(&mut app);
        app.add_systems(Update, switch_systems());
        if headset {
            app.insert_resource(Headset);
        }
        for i in 0..2 {
            app.world_mut().spawn((
                XrCamera(i),
                SplatCamera {
                    sort: SortMode::PerView,
                    output: OutputSpace::LinearOver {
                        background: Some([0.2, 0.3, 0.4]),
                    },
                },
            ));
        }
        app
    }

    #[derive(Resource, Default)]
    struct Toasts(Vec<Toast>);

    fn mode(app: &App) -> EnvironmentMode {
        app.world().resource::<Environment>().mode
    }

    fn setting(app: &App) -> String {
        app.world()
            .resource::<SettingsRes>()
            .0
            .environment
            .mode
            .clone()
    }

    fn eye_backgrounds(app: &mut App) -> Vec<Option<[f32; 3]>> {
        let mut q = app
            .world_mut()
            .query_filtered::<&SplatCamera, With<XrCamera>>();
        q.iter(app.world())
            .map(|c| match c.output {
                OutputSpace::LinearOver { background } => background,
                OutputSpace::FileEncoded => panic!("an eye draws over linear"),
            })
            .collect()
    }

    /// Q-335: on the headset, `Toggle(Feature::Passthrough)` flips `[environment] mode`,
    /// and the effective mode, the eyes' output and the request to the render world follow
    /// (Q-94, Q-234: no background in passthrough, the setting's in VR mode).
    #[test]
    fn switch_toggle_flips_the_setting_and_the_mode_follows_on_the_headset() {
        let mut app = switch_app(true, true, "vr");
        app.update();
        assert_eq!(mode(&app), EnvironmentMode::Vr);
        app.world_mut()
            .write_message(Intent::Toggle(Feature::Passthrough));
        app.update();
        assert_eq!(setting(&app), "passthrough");
        assert_eq!(mode(&app), EnvironmentMode::Passthrough);
        assert_eq!(eye_backgrounds(&mut app), vec![None, None]);
        assert_eq!(
            app.world().resource::<LayerRequest>().mode,
            EnvironmentMode::Passthrough
        );
        app.world_mut()
            .write_message(Intent::Toggle(Feature::Passthrough));
        app.update();
        assert_eq!(setting(&app), "vr");
        assert_eq!(mode(&app), EnvironmentMode::Vr);
        assert_eq!(
            eye_backgrounds(&mut app),
            vec![Some([0.2, 0.3, 0.4]), Some([0.2, 0.3, 0.4])]
        );
        assert_eq!(
            app.world().resource::<LayerRequest>().mode,
            EnvironmentMode::Vr
        );
        assert!(app.world().resource::<Toasts>().0.is_empty());
    }

    /// Q-335, §1.4 Backgrounds: without the `Headset` marker (the desktop, `play --xr`)
    /// the toggle is not consumed, the setting is kept and the mode stays VR.
    #[test]
    fn switch_without_the_headset_keeps_vr_and_the_setting() {
        let mut app = switch_app(true, false, "passthrough");
        app.world_mut()
            .write_message(Intent::Toggle(Feature::Passthrough));
        app.update();
        assert_eq!(setting(&app), "passthrough");
        assert_eq!(mode(&app), EnvironmentMode::Vr);
        assert_eq!(
            eye_backgrounds(&mut app),
            vec![Some([0.2, 0.3, 0.4]), Some([0.2, 0.3, 0.4])]
        );
        assert!(app.world().resource::<Toasts>().0.is_empty());
    }

    /// Q-335, Q-161: on a headset whose runtime lacks the extension, a setting asking for
    /// passthrough goes back to `vr` with one E-XR-UNSUPPORTED toast per session; the mode
    /// stays VR.
    #[test]
    fn switch_without_the_extension_writes_vr_back_with_one_toast_per_session() {
        let mut app = switch_app(false, true, "passthrough");
        app.update();
        assert_eq!(setting(&app), "vr");
        assert_eq!(mode(&app), EnvironmentMode::Vr);
        let toast = Toast {
            text: "E-XR-UNSUPPORTED: Not available on this runtime (XR_FB_passthrough)".to_owned(),
            level: ToastLevel::Warning,
        };
        assert_eq!(app.world().resource::<Toasts>().0, vec![toast.clone()]);
        // A second request in the session: no second toast.
        app.world_mut()
            .write_message(Intent::Toggle(Feature::Passthrough));
        app.update();
        app.update();
        assert_eq!(setting(&app), "vr");
        assert_eq!(app.world().resource::<Toasts>().0.len(), 1);
        // A new session may toast once more.
        app.world_mut().run_system_cached(new_session).unwrap();
        app.world_mut()
            .write_message(Intent::Toggle(Feature::Passthrough));
        app.update();
        app.update();
        assert_eq!(
            app.world().resource::<Toasts>().0,
            vec![toast.clone(), toast]
        );
    }

    /// The style follows the three settings (§1.9, §1.11), on `Environment` and in the
    /// render world's request.
    #[test]
    fn switch_carries_the_settings_style() {
        let mut app = switch_app(true, true, "vr");
        {
            let mut s = app.world_mut().resource_mut::<SettingsRes>();
            s.0.environment.passthrough_brightness = 0.5;
            s.0.environment.passthrough_contrast = -0.25;
            s.0.environment.passthrough_saturation = 1.0;
        }
        app.update();
        let want = PassthroughStyle {
            brightness: 0.5,
            contrast: -0.25,
            saturation: 1.0,
        };
        assert_eq!(app.world().resource::<Environment>().style, want);
        assert_eq!(app.world().resource::<LayerRequest>().style, want);
    }

    /// Q-336, §1.9 ("On the desktop the Environment tab shows the VR-mode controls
    /// only"): `XrFeatures.passthrough` is `Available` on the headset with the extension,
    /// `Absent` without either.
    #[test]
    fn switch_reports_the_feature_on_the_headset_with_the_extension() {
        for (available, headset, want) in [
            (true, true, FeatureState::Available),
            (false, true, FeatureState::Absent),
            (true, false, FeatureState::Absent),
        ] {
            let mut app = switch_app(available, headset, "vr");
            app.update();
            assert_eq!(
                app.world().resource::<XrFeatures>().passthrough,
                want,
                "available {available}, headset {headset}"
            );
        }
    }

    /// Q-336: after the session ends, `XrFeatures.passthrough` goes back to `Absent` (as
    /// occlusion's does), and the mode to VR.
    #[test]
    fn no_session_reports_the_feature_absent() {
        let mut app = switch_app(true, true, "passthrough");
        app.update();
        assert_eq!(
            app.world().resource::<XrFeatures>().passthrough,
            FeatureState::Available
        );
        app.world_mut().run_system_cached(no_session).unwrap();
        assert_eq!(
            app.world().resource::<XrFeatures>().passthrough,
            FeatureState::Absent
        );
        assert_eq!(mode(&app), EnvironmentMode::Vr);
    }

    /// §1.8: the meshes are drawn in VR mode only, and not without an `Environment`.
    #[test]
    fn devices_drawn_follows_the_environment_mode() {
        let mut app = App::new();
        app.init_resource::<DevicesDrawn>()
            .add_systems(Update, devices_drawn);
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(false));
        app.insert_resource(Environment {
            mode: EnvironmentMode::Vr,
            passthrough_available: false,
            style: PassthroughStyle::default(),
        });
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(true));
        app.world_mut().resource_mut::<Environment>().mode = EnvironmentMode::Passthrough;
        app.update();
        assert_eq!(*app.world().resource::<DevicesDrawn>(), DevicesDrawn(false));
    }
}
