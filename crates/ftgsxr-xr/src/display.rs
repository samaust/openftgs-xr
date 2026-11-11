// SPDX-License-Identifier: Apache-2.0

//! The headset display: refresh rate, fixed foveation and render scale (§1.12 Headset
//! display, D-27, §2.1 `DisplaySettings`, Q-124).
//!
//! - Refresh rate (XR\_FB\_display\_refresh\_rate): the runtime's rates go into
//!   `available_rates` once each session has begun; a wanted rate from that list is requested, and the
//!   runtime's change event is logged as `display refresh rate <from> -> <to> Hz`.
//! - Foveation (XR\_FB\_foveation and its configuration, Vulkan and swapchain-update-state
//!   companions): a profile at the setting's level is created in the main world and
//!   applied to the swapchain in the render world, where bevy\_mod\_openxr keeps it; the
//!   result is logged as `foveation <level> applied to the swapchain` or an error.
//!   bevy\_mod\_openxr creates the swapchain without a `next` chain, so the profile is
//!   applied by xrUpdateSwapchainFB only (Q-124).
//! - Render scale: a change of 0.1 or more from the scale the live swapchain was made with
//!   ends and destroys the session and creates it again (through `session`'s re-creation)
//!   with the swapchain sized at the recommended size times the scale; the crate leaks the
//!   old images. The recommended size is XR\_META\_recommended\_layer\_resolution's when the
//!   runtime gives a valid one, else the view configuration's (Q-132).
//!
//! Where an extension is absent, `available_rates` is empty, a request logs
//! E-XR-UNSUPPORTED (a warning) and the setting is kept for a later runtime (Q-124). A
//! refresh rate or a foveation asked for explicitly (a settings value, a change while the
//! session runs) also triggers a `Toast`, which ui puts on the line, at most once per
//! session per extension; the tier's default never does (Q-161).

use std::collections::HashSet;

use bevy::app::{App, Plugin, Update};
use bevy::ecs::message::MessageWriter;
use bevy::log::{info, warn};
use bevy::math::UVec2;
use bevy::prelude::DetectChangesMut;
use bevy::prelude::{Commands, IntoScheduleConfigs, Res, ResMut, Resource, resource_exists};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::{Render, RenderApp};
use bevy_mod_openxr::graphics::GraphicsWrap;
use bevy_mod_openxr::poll_events::{OxrEventHandlerExt, OxrEventIn};
use bevy_mod_openxr::resources::{
    OxrCurrentSessionConfig, OxrFrameState, OxrInstance, OxrSessionConfig, OxrSwapchain,
    OxrSystemId, OxrViews,
};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::{
    XrPostSessionBegin, XrPreDestroySession, XrRenderSystems, XrRequestExitMessage,
    XrSessionCreated, XrState,
};
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::{SettingsRes, Toast, ToastLevel};
use ftgsxr_io::Settings;
use ftgsxr_render::{Foveation, QualityTier};
use openxr::sys;
use openxr::sys::Handle as _;

use crate::XrError;
use crate::error::log_error;
use crate::session::Recreate;

mod metrics;

/// The scale change that re-creates the swapchain (§1.12).
const SCALE_STEP: f32 = 0.1;

/// The headset display's settings (§2.1, §1.12): the wanted refresh rate (`None` is the
/// runtime's), the runtime's rates, the fixed-foveation level and the render scale.
/// Changing a field is a request; `display` applies it.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct DisplaySettings {
    /// The wanted refresh rate, Hz; `None` keeps the runtime's.
    pub refresh_rate: Option<f32>,
    /// The runtime's rates, Hz; empty without XR\_FB\_display\_refresh\_rate.
    pub available_rates: Vec<f32>,
    /// The fixed-foveation level.
    pub foveation: Foveation,
    /// The swapchain's size over the recommended size.
    pub render_scale: f32,
}

impl DisplaySettings {
    /// From `XrSettings` (Q-49): `None` is the runtime's rate and the tier's foveation and
    /// render scale.
    pub(crate) fn from_settings(settings: &Settings) -> Self {
        let tier = QualityTier::by_name(&settings.general.tier).unwrap_or_else(|| {
            if cfg!(target_os = "android") {
                QualityTier::quest_balanced()
            } else {
                QualityTier::desktop()
            }
        });
        let foveation = settings
            .xr
            .foveation
            .as_deref()
            .and_then(parse_foveation)
            .unwrap_or(tier.foveation);
        DisplaySettings {
            refresh_rate: settings.xr.refresh_rate,
            available_rates: Vec::new(),
            foveation,
            render_scale: settings.xr.render_scale.unwrap_or(tier.render_scale),
        }
    }
}

/// `off`, `low`, `medium`, `high`, `dynamic` (§1.11, Q-49).
fn parse_foveation(s: &str) -> Option<Foveation> {
    Some(match s {
        "off" => Foveation::Off,
        "low" => Foveation::Low,
        "medium" => Foveation::Medium,
        "high" => Foveation::High,
        "dynamic" => Foveation::Dynamic,
        _ => return None,
    })
}

/// What the runtime offers `display`: whether each extension's table is loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Caps {
    pub(crate) refresh_rate: bool,
    pub(crate) foveation: bool,
    pub(crate) layer_resolution: bool,
}

impl Caps {
    fn of(instance: Option<&OxrInstance>) -> Caps {
        instance.map_or_else(Caps::default, |i| {
            let e = i.exts();
            Caps {
                refresh_rate: e.fb_display_refresh_rate.is_some(),
                foveation: e.fb_foveation.is_some() && e.fb_swapchain_update_state.is_some(),
                layer_resolution: e.meta_recommended_layer_resolution.is_some(),
            }
        })
    }
}

/// What to do with the wanted refresh rate.
#[derive(Debug, PartialEq)]
pub(crate) enum RateAction {
    /// Nothing wanted.
    None,
    /// Request this rate.
    Request(f32),
    /// The rate is not in the runtime's list: not requested, kept.
    NotListed(f32),
}

/// The refresh rate's decision: fills `available_rates` from the runtime's list (empty
/// without the extension) and says what to request; without the extension a wanted rate
/// is E-XR-UNSUPPORTED and stays in the settings (Q-124).
pub(crate) fn refresh_rate(
    caps: Caps,
    settings: &mut DisplaySettings,
    enumerated: Vec<f32>,
) -> Result<RateAction, XrError> {
    if !caps.refresh_rate {
        settings.available_rates.clear();
        return match settings.refresh_rate {
            Some(_) => Err(XrError::Unsupported("XR_FB_display_refresh_rate")),
            None => Ok(RateAction::None),
        };
    }
    settings.available_rates = enumerated;
    Ok(match settings.refresh_rate {
        None => RateAction::None,
        Some(r) if settings.available_rates.contains(&r) => RateAction::Request(r),
        Some(r) => RateAction::NotListed(r),
    })
}

/// The OpenXR level of a setting. `Dynamic` lets the runtime vary the level up to HIGH
/// (Q-131).
fn level(f: Foveation) -> (sys::FoveationLevelFB, sys::FoveationDynamicFB) {
    use sys::{FoveationDynamicFB as D, FoveationLevelFB as L};
    match f {
        Foveation::Off => (L::NONE, D::DISABLED),
        Foveation::Low => (L::LOW, D::DISABLED),
        Foveation::Medium => (L::MEDIUM, D::DISABLED),
        Foveation::High => (L::HIGH, D::DISABLED),
        // Q-131
        Foveation::Dynamic => (L::HIGH, D::LEVEL_ENABLED),
    }
}

/// The swapchain size for `scale` over `recommended`, within `max` (§1.12).
pub(crate) fn scaled_size(recommended: UVec2, max: UVec2, scale: f32) -> UVec2 {
    let s = |r: u32, m: u32| ((r as f32 * scale).round() as u32).clamp(1, m.max(1));
    UVec2::new(s(recommended.x, max.x), s(recommended.y, max.y))
}

/// What has been applied, so each change is acted on once.
#[derive(Resource, Debug)]
struct Applied {
    refresh_rate: Option<f32>,
    foveation: Option<Foveation>,
    /// The scale the live (or next) swapchain is made with.
    scale: f32,
    /// A requested scale that could not be applied: reported once, not every frame, and
    /// attempted again only once the setting moves or a new session is created (§1.13;
    /// Q-147).
    failed_scale: Option<f32>,
    /// Whether the settings' foveation was asked for explicitly (an absent extension is
    /// only reported for a request, Q-124).
    foveation_requested: bool,
    /// The absent extensions toasted in this session (Q-161).
    toasted: HashSet<&'static str>,
}

impl Applied {
    /// Nothing applied yet but the first swapchain's scale.
    fn new(scale: f32, foveation_requested: bool) -> Self {
        Applied {
            refresh_rate: None,
            foveation: None,
            scale,
            failed_scale: None,
            foveation_requested,
            toasted: HashSet::new(),
        }
    }

    /// Whether the absent extension `what` is to be toasted now: the first time in this
    /// session (Q-161).
    // Q-161
    fn toast_once(&mut self, what: &'static str) -> bool {
        self.toasted.insert(what)
    }

    /// Takes the render-scale change to `scale` if one is to be attempted now: it differs
    /// from the applied scale by a step, no re-creation is under way, and it is not the
    /// scale that last failed. A failure is forgotten once the setting moves off it, and at
    /// a new session.
    // Q-147
    fn take_scale_change(&mut self, scale: f32, recreating: bool) -> bool {
        if self.failed_scale.is_some_and(|f| f != scale) {
            self.failed_scale = None;
        }
        (scale - self.scale).abs() >= SCALE_STEP - 1e-6
            && !recreating
            && self.failed_scale != Some(scale)
    }

    /// Records that `scale` could not be applied, so it is not attempted again each frame.
    fn scale_failed(&mut self, scale: f32) {
        self.failed_scale = Some(scale);
    }

    /// Forgets a failed scale at a new session, whose view configuration may answer, so
    /// the setting is attempted once on it (Q-147), and the extensions toasted (Q-161).
    fn session_created(&mut self) {
        self.failed_scale = None;
        self.toasted.clear();
    }
}

/// The foveation profile the render world applies, numbered so it is applied once.
#[derive(Resource, Clone, ExtractResource, Default)]
struct FoveationRequest {
    generation: u64,
    level: Option<(Foveation, openxr::FoveationProfileFB)>,
}

/// `display`'s systems; `plugin` adds it after `SessionPlugin`.
pub(crate) struct DisplayPlugin;

impl Plugin for DisplayPlugin {
    fn build(&self, app: &mut App) {
        let settings = app
            .world()
            .get_resource::<SettingsRes>()
            .map_or_else(Settings::default, |s| s.0.clone());
        // Q-127
        let explicit_foveation = settings.xr.foveation.is_some();
        let display = DisplaySettings::from_settings(&settings);
        let scale = display.render_scale;
        if (scale - 1.0).abs() > f32::EPSILON {
            // The first session's swapchain at the scale (Q-124).
            if let Some(size) = recommended(app.world().get_resource(), app.world().get_resource())
                .map(|(r, m)| scaled_size(r, m, scale))
                && let Some(mut config) = app.world_mut().get_resource_mut::<OxrSessionConfig>()
            {
                config.resolutions = Some(vec![size]);
            }
        }
        if !app.world().contains_resource::<DisplaySettings>() {
            app.insert_resource(display);
        }
        app.insert_resource(Applied::new(scale, explicit_foveation))
            .init_resource::<FoveationRequest>()
            .add_plugins(ExtractResourcePlugin::<FoveationRequest>::default())
            .add_systems(XrSessionCreated, on_session_created)
            .add_systems(XrPostSessionBegin, on_session_begun)
            .add_systems(XrPreDestroySession, drop_profile)
            .add_systems(Update, apply_changes)
            .add_oxr_event_handler(log_rate_change);
        // #725: the runtime's counters for the device spikes (Q-T725.1).
        metrics::add(app);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(XrPreDestroySession, drop_profile);
            render_app.add_systems(
                Render,
                apply_foveation
                    .in_set(XrRenderSystems::PreRender)
                    .run_if(resource_exists::<OxrSwapchain>),
            );
        }
    }
}

/// The view configuration's recommended and maximum eye-buffer size (Appendix E.2).
fn recommended(
    instance: Option<&OxrInstance>,
    system: Option<&OxrSystemId>,
) -> Option<(UVec2, UVec2)> {
    let views = instance?
        .enumerate_view_configuration_views(
            system?.0,
            openxr::ViewConfigurationType::PRIMARY_STEREO,
        )
        .ok()?;
    let v = views.first()?;
    Some((
        UVec2::new(
            v.recommended_image_rect_width,
            v.recommended_image_rect_height,
        ),
        UVec2::new(v.max_image_rect_width, v.max_image_rect_height),
    ))
}

/// Logs an E-XR-UNSUPPORTED (§1.15: the control is hidden, so a warning).
// Q-134
fn unsupported(e: &XrError) {
    warn!("{}: {}", e.code(), e.user_text());
}

/// An E-XR-UNSUPPORTED for a feature asked for explicitly: the warning each time and a
/// `Toast` triggered for ui the first time in the session for its extension (Q-161; every
/// caller's request is explicit, the tier's default foveation and an unset refresh rate
/// never reach it).
// Q-161
fn unsupported_requested(e: &XrError, applied: &mut Applied, commands: &mut Commands) {
    unsupported(e);
    if let XrError::Unsupported(what) = e
        && !applied.toast_once(what)
    {
        return;
    }
    commands.trigger(Toast {
        text: format!("{}: {}", e.code(), e.user_text()),
        level: ToastLevel::Warning,
    });
}

/// At each session: the rates, the wanted rate, the foveation again (the swapchain is
/// new) and the swapchain's size; a failed render scale is forgotten (Q-147).
fn on_session_created(
    session: Option<Res<OxrSession>>,
    instance: Option<Res<OxrInstance>>,
    current: Option<Res<OxrCurrentSessionConfig>>,
    settings: Res<DisplaySettings>,
    mut applied: ResMut<Applied>,
    mut request: ResMut<FoveationRequest>,
    mut commands: Commands,
) {
    let Some(session) = session else { return };
    // Q-147
    applied.session_created();
    if let Some(c) = current {
        info!(
            "swapchain {}x{} at render scale {} (§1.12)",
            c.resolution.x, c.resolution.y, applied.scale
        );
    }
    let caps = Caps::of(instance.as_deref());
    let f = settings.foveation;
    if caps.foveation || (f != Foveation::Off && applied.foveation_requested) {
        foveation(caps, &session, f, &mut request, &mut applied, &mut commands);
    }
    applied.foveation = Some(f);
}

/// Once the session has begun: the rates and the wanted rate (the request is made on a
/// running session).
fn on_session_begun(
    session: Option<Res<OxrSession>>,
    instance: Option<Res<OxrInstance>>,
    mut settings: ResMut<DisplaySettings>,
    mut applied: ResMut<Applied>,
    mut commands: Commands,
) {
    let Some(session) = session else { return };
    apply_rate(
        Caps::of(instance.as_deref()),
        &session,
        &mut settings,
        &mut applied,
        &mut commands,
    );
    applied.refresh_rate = settings.refresh_rate;
}

/// Drops the foveation profile with its session (it is a child of the session).
fn drop_profile(request: Option<ResMut<FoveationRequest>>) {
    if let Some(mut r) = request {
        r.level = None;
    }
}

/// Enumerates the rates and requests the wanted one.
fn apply_rate(
    caps: Caps,
    session: &OxrSession,
    settings: &mut DisplaySettings,
    applied: &mut Applied,
    commands: &mut Commands,
) {
    let enumerated = if caps.refresh_rate {
        // The wrapper panics without the extension; `caps` checked it is loaded.
        session
            .enumerate_display_refresh_rates()
            .unwrap_or_else(|e| {
                log_error(&XrError::Runtime(format!("refresh rates: {e}")));
                Vec::new()
            })
    } else {
        Vec::new()
    };
    match refresh_rate(caps, settings, enumerated) {
        Err(e) => unsupported_requested(&e, applied, commands),
        Ok(RateAction::None) => {}
        Ok(RateAction::NotListed(r)) => warn!(
            "refresh rate {r} Hz is not among the runtime's {:?}: kept, not requested",
            settings.available_rates
        ),
        Ok(RateAction::Request(r)) => match session.request_display_refresh_rate(r) {
            Ok(()) => info!("display refresh rate {r} Hz requested"),
            Err(e) => log_error(&XrError::Runtime(format!("refresh rate {r} Hz: {e}"))),
        },
    }
}

/// Creates the profile for `f` and hands it to the render world, or reports the
/// extension absent.
fn foveation(
    caps: Caps,
    session: &OxrSession,
    f: Foveation,
    request: &mut FoveationRequest,
    applied: &mut Applied,
    commands: &mut Commands,
) {
    if !caps.foveation {
        unsupported_requested(&XrError::Unsupported("XR_FB_foveation"), applied, commands);
        return;
    }
    let (level, dynamic) = level(f);
    match session.create_foveation_profile(Some(openxr::FoveationLevelProfile {
        level,
        vertical_offset: 0.0,
        dynamic,
    })) {
        Ok(profile) => {
            request.generation += 1;
            request.level = Some((f, profile));
        }
        Err(e) => log_error(&XrError::Runtime(format!("foveation {f:?}: {e}"))),
    }
}

/// The main world's part of a change to `DisplaySettings` while the session runs: each
/// field is compared with what was applied, not with Bevy's change detection, so a change
/// made while the session is not running is applied once it runs (§1.12).
#[allow(clippy::too_many_arguments)]
fn apply_changes(
    session: Option<Res<OxrSession>>,
    instance: Option<Res<OxrInstance>>,
    system: Option<Res<OxrSystemId>>,
    state: Option<Res<XrState>>,
    mut settings: ResMut<DisplaySettings>,
    mut applied: ResMut<Applied>,
    mut request: ResMut<FoveationRequest>,
    layer: LayerInputs,
    mut config: ResMut<OxrSessionConfig>,
    mut recreate: ResMut<Recreate>,
    mut exit: MessageWriter<XrRequestExitMessage>,
    mut commands: Commands,
) {
    if state.as_deref() != Some(&XrState::Running) {
        return;
    }
    let Some(session) = session else { return };
    let caps = Caps::of(instance.as_deref());
    if settings.refresh_rate != applied.refresh_rate {
        applied.refresh_rate = settings.refresh_rate;
        let s = settings.bypass_change_detection();
        apply_rate(caps, &session, s, &mut applied, &mut commands);
    }
    if Some(settings.foveation) != applied.foveation {
        applied.foveation = Some(settings.foveation);
        applied.foveation_requested = true;
        foveation(
            caps,
            &session,
            settings.foveation,
            &mut request,
            &mut applied,
            &mut commands,
        );
    }
    let scale = settings.render_scale;
    if applied.take_scale_change(scale, recreate.0) {
        let Some((view, max)) = recommended(instance.as_deref(), system.as_deref()) else {
            applied.scale_failed(scale);
            log_error(&XrError::Runtime(format!(
                "render scale {scale}: no view configuration; not applied"
            )));
            return;
        };
        // Q-132
        let base = if caps.layer_resolution {
            layer_resolution(&session, &instance, &layer).unwrap_or(view)
        } else {
            unsupported(&XrError::Unsupported(
                "XR_META_recommended_layer_resolution",
            ));
            view
        };
        let size = scaled_size(base, max, scale);
        info!(
            "render scale {} -> {scale}: re-creating the session with a {}x{} swapchain \
             (the old swapchain's images are leaked by bevy_mod_openxr)",
            applied.scale, size.x, size.y
        );
        applied.scale = scale;
        config.resolutions = Some(vec![size]);
        recreate.0 = true;
        exit.write(XrRequestExitMessage);
    }
}

/// What XR\_META\_recommended\_layer\_resolution's query needs from the main world.
type LayerInputs<'w> = (
    Option<Res<'w, OxrViews>>,
    Option<Res<'w, OxrFrameState>>,
    Option<Res<'w, XrPrimaryReferenceSpace>>,
);

/// The runtime's recommendation for a new projection layer's swapchain (null swapchain
/// handles, as the extension provides for an allocation), when valid (Q-132).
fn layer_resolution(
    session: &OxrSession,
    instance: &Option<Res<OxrInstance>>,
    (views, frame, space): &LayerInputs,
) -> Option<UVec2> {
    let fp = instance
        .as_ref()?
        .exts()
        .meta_recommended_layer_resolution?;
    let (views, frame, space) = (views.as_ref()?, frame.as_ref()?, space.as_ref()?);
    let proj_views: Vec<sys::CompositionLayerProjectionView> = views
        .iter()
        .enumerate()
        .map(|(i, v)| sys::CompositionLayerProjectionView {
            ty: sys::CompositionLayerProjectionView::TYPE,
            next: std::ptr::null(),
            pose: v.pose,
            fov: v.fov,
            sub_image: sys::SwapchainSubImage {
                swapchain: sys::Swapchain::NULL,
                image_rect: sys::Rect2Di::default(),
                image_array_index: i as u32,
            },
        })
        .collect();
    let layer = sys::CompositionLayerProjection {
        ty: sys::CompositionLayerProjection::TYPE,
        next: std::ptr::null(),
        layer_flags: sys::CompositionLayerFlags::EMPTY,
        space: sys::Space::from_raw(space.0.0.as_raw()),
        view_count: proj_views.len() as u32,
        views: proj_views.as_ptr(),
    };
    let info = sys::RecommendedLayerResolutionGetInfoMETA {
        ty: sys::RecommendedLayerResolutionGetInfoMETA::TYPE,
        next: std::ptr::null(),
        layer: (&raw const layer).cast(),
        predicted_display_time: frame.predicted_display_time,
    };
    let mut out = sys::RecommendedLayerResolutionMETA::out(std::ptr::null_mut());
    // SAFETY: the extension's table is loaded (checked above, so the pointer is the
    // runtime's); the session handle is alive (a main-world resource while the session
    // exists); `info`, `layer` and `proj_views` are fully initialised with their TYPEs
    // and outlive the call; `out` is the extension's output struct with its TYPE set.
    let res =
        unsafe { (fp.get_recommended_layer_resolution)(session.as_raw(), &info, out.as_mut_ptr()) };
    if res.into_raw() < 0 {
        log_error(&XrError::Runtime(format!(
            "recommended layer resolution: {res:?}"
        )));
        return None;
    }
    // SAFETY: the call succeeded, so the runtime filled `out`.
    let out = unsafe { out.assume_init() };
    if out.is_valid.into() {
        let d = out.recommended_image_dimensions;
        info!("recommended layer resolution {}x{}", d.width, d.height);
        Some(UVec2::new(d.width.max(1) as u32, d.height.max(1) as u32))
    } else {
        info!("recommended layer resolution: none given; the view configuration's used");
        None
    }
}

/// Applies the latest profile to the swapchain once (render world).
fn apply_foveation(
    request: Res<FoveationRequest>,
    instance: Res<OxrInstance>,
    swapchain: Res<OxrSwapchain>,
    mut done: bevy::prelude::Local<u64>,
) {
    if request.generation == *done {
        return;
    }
    *done = request.generation;
    let Some((f, profile)) = &request.level else {
        return;
    };
    let Some(fp) = instance.exts().fb_swapchain_update_state else {
        unsupported(&XrError::Unsupported("XR_FB_swapchain_update_state"));
        return;
    };
    let handle = match &swapchain.0 {
        GraphicsWrap::Vulkan(s) => s.as_raw(),
    };
    let state = sys::SwapchainStateFoveationFB {
        ty: sys::SwapchainStateFoveationFB::TYPE,
        next: std::ptr::null_mut(),
        flags: sys::SwapchainStateFoveationFlagsFB::EMPTY,
        profile: profile.as_raw(),
    };
    // SAFETY: the extension's table is loaded (checked above); the swapchain handle is
    // alive (the render world's `OxrSwapchain`, removed before the session is
    // destroyed); `state` is fully initialised with its TYPE and flags 0, and is read
    // through its base header as the extension defines; `profile` is held by
    // `FoveationRequest` for the call and after.
    let res = unsafe { (fp.update_swapchain)(handle, (&raw const state).cast()) };
    if res.into_raw() < 0 {
        log_error(&XrError::Runtime(format!(
            "foveation {f:?} on the swapchain: {res:?}"
        )));
    } else {
        info!("foveation {f:?} applied to the swapchain");
    }
}

/// Logs the runtime's refresh-rate change (the rate took effect).
fn log_rate_change(event: OxrEventIn) {
    if let openxr::Event::DisplayRefreshRateChangedFB(e) = *event {
        info!(
            "display refresh rate {} -> {} Hz",
            e.from_display_refresh_rate(),
            e.to_display_refresh_rate()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A render scale that fails is attempted (and so logged) once per requested change,
    /// not every frame, and again once the setting moves or a new session is created
    /// (§1.13; Q-147). `apply_changes` itself needs a running session, so the test is of
    /// the decision it takes each frame.
    #[test]
    fn take_scale_change_skips_a_failed_scale_until_the_setting_moves() {
        let mut applied = Applied::new(1.0, false);
        assert!(!applied.take_scale_change(1.0, false));
        assert!(applied.take_scale_change(1.5, false));
        assert!(!applied.take_scale_change(1.5, true));
        applied.scale_failed(1.5);
        for _ in 0..3 {
            assert!(!applied.take_scale_change(1.5, false));
        }
        assert!(applied.take_scale_change(1.2, false));
        assert!(applied.take_scale_change(1.5, false));
        // A new session forgets the failure: the unchanged setting is attempted once on
        // it, then skipped again (Q-147, owner #272).
        applied.scale_failed(1.5);
        assert!(!applied.take_scale_change(1.5, false));
        applied.session_created();
        assert!(applied.take_scale_change(1.5, false));
        applied.scale_failed(1.5);
        assert!(!applied.take_scale_change(1.5, false));
    }

    /// An absent extension asked for explicitly is toasted at most once per session, each
    /// extension apart; a new session may toast it once more (Q-161, owner #306).
    #[test]
    fn toast_once_is_once_per_session_per_extension() {
        let mut applied = Applied::new(1.0, true);
        assert!(applied.toast_once("XR_FB_foveation"));
        assert!(!applied.toast_once("XR_FB_foveation"));
        assert!(applied.toast_once("XR_FB_display_refresh_rate"));
        assert!(!applied.toast_once("XR_FB_display_refresh_rate"));
        assert!(!applied.toast_once("XR_FB_foveation"));
        applied.session_created();
        assert!(applied.toast_once("XR_FB_foveation"));
        assert!(!applied.toast_once("XR_FB_foveation"));
        assert!(applied.toast_once("XR_FB_display_refresh_rate"));
    }

    fn wanting(rate: Option<f32>) -> DisplaySettings {
        DisplaySettings {
            refresh_rate: rate,
            available_rates: vec![72.0],
            foveation: Foveation::Off,
            render_scale: 1.0,
        }
    }

    // The no-extension fallback (Q-124's detecting test, owner 2026-11-02, #228).
    #[test]
    fn refresh_rate_without_the_extension_is_unsupported_and_kept() {
        let mut s = wanting(Some(90.0));
        let r = refresh_rate(Caps::default(), &mut s, vec![72.0, 90.0]);
        assert_eq!(r, Err(XrError::Unsupported("XR_FB_display_refresh_rate")));
        assert_eq!(r.unwrap_err().code(), "E-XR-UNSUPPORTED");
        assert!(s.available_rates.is_empty());
        assert_eq!(s.refresh_rate, Some(90.0));
        let mut none = wanting(None);
        assert_eq!(
            refresh_rate(Caps::default(), &mut none, vec![]),
            Ok(RateAction::None)
        );
        assert!(none.available_rates.is_empty());
    }

    #[test]
    fn refresh_rate_with_the_extension_lists_and_requests() {
        let caps = Caps {
            refresh_rate: true,
            ..Caps::default()
        };
        let quest = vec![72.0, 80.0, 90.0, 96.0, 100.0, 120.0];
        let mut s = wanting(Some(90.0));
        assert_eq!(
            refresh_rate(caps, &mut s, quest.clone()),
            Ok(RateAction::Request(90.0))
        );
        assert_eq!(s.available_rates, quest);
        let mut odd = wanting(Some(60.0));
        assert_eq!(
            refresh_rate(caps, &mut odd, quest),
            Ok(RateAction::NotListed(60.0))
        );
        assert_eq!(odd.refresh_rate, Some(60.0));
    }

    #[test]
    fn scaled_size_rounds_and_clamps() {
        let rec = UVec2::new(1680, 1760);
        assert_eq!(
            scaled_size(rec, UVec2::new(4096, 4096), 0.8),
            UVec2::new(1344, 1408)
        );
        assert_eq!(
            scaled_size(rec, UVec2::new(2000, 2000), 1.5),
            UVec2::new(2000, 2000)
        );
    }

    #[test]
    fn from_settings_takes_the_tier_for_none() {
        let mut s = Settings::default();
        s.general.tier = "quest-low".into();
        let d = DisplaySettings::from_settings(&s);
        assert_eq!(
            (d.foveation, d.render_scale, d.refresh_rate),
            (Foveation::High, 0.8, None)
        );
        s.xr.foveation = Some("low".into());
        s.xr.render_scale = Some(0.9);
        s.xr.refresh_rate = Some(90.0);
        let d = DisplaySettings::from_settings(&s);
        assert_eq!(
            (d.foveation, d.render_scale, d.refresh_rate),
            (Foveation::Low, 0.9, Some(90.0))
        );
    }
}
