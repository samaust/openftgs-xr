// SPDX-License-Identifier: Apache-2.0

//! The light estimator's control (§1.9, E14, D-15, D-38): the main world's `Update`,
//! after `tracking`.
//!
//! - The feature and the permission (Q-384): `XrFeatures.lighting` is `Absent` without a
//!   session, without the platform's `CameraFrames`, or once `start` reported no
//!   passthrough camera (Q-392); `Denied` after a denial, or after `start` reported a
//!   missing camera permission (Q-392); `Available` otherwise; any other failed start is
//!   retried 2 s later at the earliest (Q-392).
//!   `Permission::Camera` is asked the first time an auto mode is wanted where the
//!   camera would then run, the session running in passthrough mode (Q-348's pattern); a denial writes `lighting_mode = manual`, toasts E-XR-PERMISSION with Y
//!   "light estimation" (Q-261) and lists the permission (Q-322); an auto mode selected
//!   again or `AskPermissionAgain(Permission::Camera)` asks once more (Q-355). The auto
//!   mode a denial wrote back is kept as session state and published as
//!   `Lighting.denied_auto`, which the Environment tab's "Ask again" writes back into the
//!   setting, so a grant runs it with no second click; a grant clears it (Q-384, Q-395).
//! - The camera's life and the estimate (Q-385): the camera runs while the effective
//!   lighting mode is auto, the permission is granted, the session runs and the effective
//!   environment mode is passthrough; the smoother restarts at each start; no new
//!   thumbnail for 2 s while started counts as a stop. Each new thumbnail is pushed into
//!   the smoother. The warm-up, restarted with the smoother, is over once two thumbnails
//!   have arrived and the newest one's `at_s` is 3τ after the first one's. While
//!   `lighting_reference` is `None`, or below the floor of 1/255 or not finite, the
//!   smoothed estimate is written into it at the first new thumbnail after the warm-up
//!   where the estimate passes the floor, `Lighting.too_dark` set while it waits for the
//!   floor. In the auto modes `Lighting.transform` is the manual form of
//!   `lighting_manual` times E14's gains (E14, revision 52), the manual form alone where
//!   the gains are not in force (VR mode, during the warm-up and while the capture
//!   waits); `HudData.light_gains` holds E14's part alone while the camera runs (Q-389,
//!   Q-394).

use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use bevy::app::{App, Plugin, Update};
use bevy::camera::Projection;
use bevy::ecs::message::MessageReader;
use bevy::log::{info, warn};
use bevy::math::{URect, Vec3};
use bevy::prelude::{
    Commands, Entity, GlobalTransform, Has, IntoScheduleConfigs, Query, Res, ResMut, Resource,
};
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::{XrPreDestroySession, XrSessionCreated, XrState};
use ftgsxr_app::{
    AskPermissionAgain, CameraFrames, DeniedPermissions, FeatureState, HudData, Lighting,
    Permission, Platform, PlatformRes, Selected, SettingsRes, Toast, ToastLevel, XrFeatures,
};
use ftgsxr_io::EnvironmentSettings;
use ftgsxr_render::ColorTransform;
use ftgsxr_render::{LayerProbe, LayerSums, SplatBounds};

use super::exposure::box_cells;

use super::band::{DepthReadback, DepthSnapshot, band_instance, thumb_depths};
use super::{
    ExposureMatch, LightEstimate, LightSmoother, LightingMode, band_mask, light_gains, linear_luma,
    manual_gains, matched_exposure_ev,
};
use crate::error::{XrError, log_warning};
use crate::passthrough::{Environment, EnvironmentMode};
use crate::tracking::XrHands;
use crate::tracking::hands::{Flow, denied_text};

/// E14's band half-width Δ.
const BAND_DELTA_M: f32 = 0.5;
/// E14's time constant.
const TAU_S: f32 = 0.5;
/// Q-385, Q-392: no new thumbnail for this long while started counts as a stop, and a
/// failed start is retried this long after at the earliest.
const STALE: Duration = Duration::from_secs(2);
/// What E-XR-UNSUPPORTED names when the shim lists no passthrough camera (Q-392).
const UNSUPPORTED: &str = "passthrough camera";
/// Q-261's feature Y.
const FEATURE: &str = "light estimation";

/// Q-385: whether the camera runs: an auto mode in force, the permission granted, the
/// session running and the effective environment mode passthrough.
// Q-385
pub(crate) fn wanted(
    mode: &LightingMode,
    granted: bool,
    running: bool,
    env: EnvironmentMode,
) -> bool {
    mode.is_auto() && granted && running && env == EnvironmentMode::Passthrough
}

/// Q-385: the warm-up is over once two thumbnails have arrived since the start and the
/// newest one's `at_s` is at least 3τ after the first one's.
// Q-385
fn warmed_up(warmup: Option<(f64, f64, u32)>) -> bool {
    warmup.is_some_and(|(first, newest, n)| n >= 2 && newest - first >= f64::from(3.0 * TAU_S))
}

/// Q-385, Q-386: the reference's floor, one code value of the thumbnail's gamma-encoded
/// full-range values.
// Q-385, Q-386
const REFERENCE_FLOOR: f32 = 1.0 / 255.0;

/// Q-385: Ȳ, r̄, ḡ and b̄ all finite and at least [`REFERENCE_FLOOR`], so Q-382's ratios against
/// the estimate as a reference stay off the clamps.
// Q-382, Q-385, Q-386
fn passes_floor(e: &LightEstimate) -> bool {
    std::iter::once(e.luma)
        .chain(e.rgb)
        .all(|v| v.is_finite() && v >= REFERENCE_FLOOR)
}

/// What one frame of the control reads.
pub(crate) struct Inputs<'a> {
    /// `[environment]`.
    pub(crate) env: &'a EnvironmentSettings,
    /// The effective environment mode.
    pub(crate) env_mode: EnvironmentMode,
    /// The session is running.
    pub(crate) running: bool,
    /// `AskPermissionAgain(Permission::Camera)` arrived.
    pub(crate) asked_again: bool,
    /// The wall clock.
    pub(crate) now: Instant,
    /// Auto local's band: the depth while the provider runs and an instance is selected.
    pub(crate) band: Option<Band<'a>>,
    /// E14's exposure match (Q-396).
    pub(crate) exposure: Exposure<'a>,
}

/// What the exposure match reads in a frame (E14, Q-396).
#[derive(Clone, Copy, Default)]
pub(crate) struct Exposure<'a> {
    /// `Lighting.match_exposure`: the Environment tab's press.
    pub(crate) pressed: bool,
    /// `LayerProbe.result`, arrived this frame.
    pub(crate) probe: Option<&'a LayerSums>,
    /// The measured cells: "per instance"'s target's projected B_m, or all when `None`.
    pub(crate) cells: Option<URect>,
}

/// Auto local's depth and the selected instance's distance d_m from the head (Q-388).
#[derive(Clone, Copy)]
pub(crate) struct Band<'a> {
    pub(crate) depth: &'a DepthSnapshot,
    pub(crate) d_m: f32,
}

/// What one frame of the control writes.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Outputs {
    /// `XrFeatures.lighting`.
    pub(crate) feature: FeatureState,
    /// `Lighting.transform`, in an effective auto mode only: the manual form times E14's
    /// gains, or the manual form alone where the gains are not in force (Q-383).
    pub(crate) transform: Option<ColorTransform>,
    /// `HudData.light_gains`: E14's part alone (Q-389, Q-394).
    pub(crate) light_gains: Option<Vec3>,
    /// `lighting_mode` written back (Q-355, Q-392).
    pub(crate) write_mode: Option<&'static str>,
    /// `lighting_reference` captured (Q-385).
    pub(crate) write_reference: Option<[f32; 4]>,
    /// `Lighting.too_dark` (Q-385).
    pub(crate) too_dark: bool,
    /// Toasts raised.
    pub(crate) toasts: Vec<String>,
    /// `Permission::Camera` listed in (true) or out of (false) `DeniedPermissions`.
    pub(crate) listed: Option<bool>,
    /// `Lighting.denied_auto`: the auto mode the last denial wrote back (Q-395).
    pub(crate) denied_auto: Option<&'static str>,
    /// `Lighting.can_match` (Q-396).
    pub(crate) can_match: bool,
    /// `Lighting.match_exposure` cleared (Q-396).
    pub(crate) consume_match: bool,
    /// `LayerProbe.request` set (Q-396).
    pub(crate) request_probe: bool,
    /// `lighting_manual.exposure_ev` written by the match (Q-396).
    pub(crate) write_exposure: Option<f32>,
    /// Information toasts raised (the match's, Q-396).
    pub(crate) notes: Vec<String>,
}

/// The control's state, kept across sessions (the permission flow) and within one (the
/// camera, the absent camera, the toast).
#[derive(Resource)]
pub(crate) struct Control {
    pub(crate) flow: Flow,
    /// The platform's camera, looked up once.
    camera: Option<Option<Arc<dyn CameraFrames>>>,
    /// A session exists.
    pub(crate) session: bool,
    /// `start` said no passthrough camera in this session (Q-392).
    unsupported: bool,
    /// E-XR-UNSUPPORTED's toast raised in this session (Q-161).
    toasted: bool,
    /// The setting was an auto mode last frame (Q-355's re-selection).
    last_auto: bool,
    /// The auto mode the last denial wrote back to `manual`, until a grant; kept across
    /// sessions with the flow (Q-384, Q-395).
    denied_auto: Option<&'static str>,
    /// The camera is started: since when a thumbnail was last new.
    started: Option<Instant>,
    /// A failed start is retried from then on (Q-392).
    retry_after: Option<Instant>,
    /// The last thumbnail's `at_s`.
    pub(crate) last_at_s: Option<f64>,
    pub(crate) smoother: LightSmoother,
    /// Q-385's warm-up since the start: the first pushed thumbnail's `at_s`, the newest
    /// one's and how many were pushed.
    warmup: Option<(f64, f64, u32)>,
    /// The last new thumbnail's raw estimate (Dv-11).
    pub(crate) raw: Option<LightEstimate>,
    /// The new thumbnails since the plugin started (Dv-11).
    pub(crate) thumbs: u64,
    /// A thumbnail carried the camera (Dv-11).
    pub(crate) pose_seen: bool,
    /// E14's L_room of the newest thumbnail since the start, over the mode's Ω.
    room: Option<f32>,
    /// A layer probe was requested for the match and its result is awaited (Q-396).
    awaiting: bool,
}

impl Default for Control {
    fn default() -> Control {
        Control {
            flow: Flow::Idle,
            camera: None,
            session: false,
            unsupported: false,
            toasted: false,
            last_auto: false,
            denied_auto: None,
            started: None,
            retry_after: None,
            last_at_s: None,
            smoother: LightSmoother::new(TAU_S),
            warmup: None,
            raw: None,
            thumbs: 0,
            pose_seen: false,
            room: None,
            awaiting: false,
        }
    }
}

/// Asks for `Permission::Camera`; without a platform it counts as granted (Q-348).
fn ask(platform: Option<&dyn Platform>) -> Flow {
    match platform {
        Some(p) => Flow::Asking(p.request_permission(Permission::Camera)),
        None => Flow::Granted,
    }
}

impl Control {
    /// The camera is started (Dv-11).
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) fn running(&self) -> bool {
        self.started.is_some()
    }

    fn feature(&self) -> FeatureState {
        let camera = self.camera.as_ref().is_some_and(Option::is_some);
        if !self.session || !camera || self.unsupported {
            FeatureState::Absent
        } else if self.flow == Flow::Denied {
            FeatureState::Denied
        } else {
            FeatureState::Available
        }
    }

    fn stop(&mut self) {
        if self.started.take().is_some()
            && let Some(Some(c)) = &self.camera
        {
            c.stop();
            info!("light: the camera stopped");
        }
    }

    /// A denial of `Permission::Camera` (Q-384, Q-261, Q-322, Q-355, Q-392): `Denied`,
    /// listed, toasted, and an auto setting written back to `manual`, its mode kept for
    /// the Environment tab's "Ask again" (Q-395).
    fn deny(&mut self, out: &mut Outputs, setting: &mut String, auto: bool) {
        self.flow = Flow::Denied;
        out.listed = Some(true);
        if auto {
            // Q-395: Q-50's spellings, auto local or else auto global.
            self.denied_auto = Some(if setting.as_str() == "auto-local" {
                "auto-local"
            } else {
                "auto-global"
            });
            *setting = "manual".into();
            out.write_mode = Some("manual");
        }
        let text = denied_text(Permission::Camera, FEATURE);
        warn!("{text}");
        out.toasts.push(text);
    }

    /// One frame (Q-384, Q-385, Q-389, Q-392).
    // Q-161, Q-261, Q-322, Q-348, Q-355, Q-384, Q-385, Q-389, Q-392, Q-394, Q-395
    pub(crate) fn step(&mut self, platform: Option<&dyn Platform>, i: &Inputs) -> Outputs {
        let mut out = Outputs::default();
        if self.camera.is_none() {
            self.camera = Some(platform.and_then(|p| p.camera_frames()));
        }
        let mut setting = i.env.lighting_mode.clone();
        // The setting's own mode, as if the camera were available (Q-50's spellings).
        let auto_setting = |s: &str| {
            let mut e = i.env.clone();
            e.lighting_mode = s.to_owned();
            LightingMode::from_settings(&e, FeatureState::Available).is_auto()
        };

        // Q-355: while denied, an auto mode selected again or "ask again" asks once more.
        let reselected = auto_setting(&setting) && !self.last_auto;
        if self.flow == Flow::Denied && (i.asked_again || reselected) {
            self.flow = ask(platform);
            out.listed = Some(false);
        }
        // Q-348, Q-384: asked the first time an auto mode is wanted where the camera
        // would then run (available, the session running, passthrough mode).
        if auto_setting(&setting)
            && i.running
            && i.env_mode == EnvironmentMode::Passthrough
            && self.feature() == FeatureState::Available
            && self.flow == Flow::Idle
        {
            self.flow = ask(platform);
        }
        if let Flow::Asking(h) = self.flow
            && let Some(p) = platform
        {
            match p.poll_permission(h) {
                Poll::Pending => {}
                Poll::Ready(true) => self.flow = Flow::Granted,
                Poll::Ready(false) => {
                    let auto = auto_setting(&setting);
                    self.deny(&mut out, &mut setting, auto);
                }
            }
        }

        let mut mode = self.mode_of(i.env, &setting);
        let run = wanted(&mode, self.flow == Flow::Granted, i.running, i.env_mode);

        // The camera's life.
        if self.started.is_some() && !run {
            self.stop();
        }
        if let Some(last) = self.started
            && i.now.duration_since(last) >= STALE
        {
            warn!(
                "light: no camera thumbnail for {} s; the camera is stopped",
                STALE.as_secs()
            );
            self.stop();
        }
        let camera = self.camera.clone().flatten();
        if run
            && self.started.is_none()
            && self.retry_after.is_none_or(|t| i.now >= t)
            && let Some(c) = &camera
        {
            match c.start() {
                Ok(()) => {
                    info!("light: the camera started");
                    self.started = Some(i.now);
                    self.retry_after = None;
                    self.smoother.reset();
                    self.warmup = None;
                    self.last_at_s = None;
                    self.room = None;
                }
                Err(e) if e.starts_with(XrError::Unsupported("").code()) => {
                    self.unsupported = true;
                    let err = XrError::Unsupported(UNSUPPORTED);
                    log_warning(&err);
                    if !self.toasted {
                        self.toasted = true;
                        out.toasts
                            .push(format!("{}: {}", err.code(), err.user_text()));
                    }
                    setting = "manual".into();
                    out.write_mode = Some("manual");
                    mode = self.mode_of(i.env, &setting);
                }
                // Q-392: a permission the shim finds missing after a grant (one of the
                // two Android permissions, or one revoked) is a denial, not retried.
                Err(e) if e.starts_with(XrError::Permission(Permission::Camera).code()) => {
                    warn!("light: {e}");
                    self.deny(&mut out, &mut setting, true);
                    mode = self.mode_of(i.env, &setting);
                }
                Err(e) => {
                    warn!("light: {e}; retried in {} s", STALE.as_secs());
                    self.retry_after = Some(i.now + STALE);
                }
            }
        }

        // The estimate.
        let mut new = false;
        if self.started.is_some()
            && let Some(c) = &camera
            && let Some(thumb) = c.latest()
            && self.last_at_s != Some(thumb.at_s)
        {
            self.last_at_s = Some(thumb.at_s);
            self.started = Some(i.now);
            self.thumbs += 1;
            self.pose_seen |= thumb.camera.is_some();
            // E14 local (Q-388): the band while the depth and a selection exist and
            // enough pixels qualify, else the frame.
            let mask = match (&mode, i.band) {
                (LightingMode::AutoLocal, Some(b)) => {
                    band_mask(&thumb_depths(&thumb, b.depth), b.d_m, BAND_DELTA_M)
                }
                _ => None,
            };
            // E14's exposure match: L_room over the same Ω.
            self.room = linear_luma(&thumb, mask.as_deref());
            if let Some(e) = LightEstimate::from_thumb(&thumb, mask.as_deref()) {
                self.raw = Some(e);
                self.smoother.push(e, thumb.at_s);
                self.warmup = Some(match self.warmup {
                    None => (thumb.at_s, thumb.at_s, 1),
                    Some((first, _, n)) => (first, thumb.at_s, n.saturating_add(1)),
                });
                new = true;
            }
        }

        // Q-385: the capture waits for the warm-up and the floor; a reference below the
        // floor reads as `None`; E14's gains are not in force until the warm-up is over,
        // nor while the capture waits.
        let auto = mode.is_auto();
        let warm = self.started.is_some() && warmed_up(self.warmup);
        let mut reference = i
            .env
            .lighting_reference
            .map(|[luma, r, g, b]| LightEstimate {
                luma,
                rgb: [r, g, b],
            })
            .filter(passes_floor);
        let estimate = self.smoother.current();
        let usable = estimate.filter(passes_floor);
        if auto
            && reference.is_none()
            && new
            && warm
            && let Some(e) = usable
        {
            reference = Some(e);
            out.write_reference = Some([e.luma, e.rgb[0], e.rgb[1], e.rgb[2]]);
        }
        out.too_dark = auto && reference.is_none() && warm && usable.is_none();
        // E14's gains where they are in force, `None` where they are not.
        let e14 = match (auto, i.env_mode, estimate, reference) {
            (true, EnvironmentMode::Passthrough, Some(e), Some(r)) if warm => {
                Some(light_gains(&e, &r, i.env.lighting_gamma))
            }
            _ => None,
        };
        if auto {
            // E14 (revision 52, #689): the manual form times E14's gains per channel,
            // the product unclamped, or the manual form alone where the gains are not
            // in force; read each frame, so a slider change applies at the next.
            // Q-383, Q-385
            let manual = manual_gains(&i.env.lighting_manual);
            out.transform = Some(ColorTransform {
                gain: manual.gain * e14.map_or(Vec3::ONE, |t| t.gain),
            });
        }
        // Q-389, Q-394: E14's part alone, without the manual offset.
        out.light_gains = self.started.map(|_| e14.map_or(Vec3::ONE, |t| t.gain));
        self.last_auto = auto_setting(&setting);
        out.feature = self.feature();
        // Q-395: a grant ends the kept mode.
        if self.flow == Flow::Granted {
            self.denied_auto = None;
        }
        out.denied_auto = self.denied_auto;
        self.exposure_match(&mut out, i, auto && warm);
        out
    }

    /// E14's exposure match (Q-396): `can_match` while an auto mode runs the camera past
    /// its warm-up with a thumbnail; a press consumed at once, requesting one probe when
    /// matching is available; the awaited result written into the manual exposure alone.
    // Q-396, Q-397
    fn exposure_match(&mut self, out: &mut Outputs, i: &Inputs, warm_auto: bool) {
        out.can_match = warm_auto && self.room.is_some();
        if self.awaiting
            && let Some(sums) = i.exposure.probe
        {
            self.awaiting = false;
            let room = self.room.unwrap_or(f32::NAN);
            let current = i.env.lighting_manual.exposure_ev;
            let note = match matched_exposure_ev(current, room, sums, i.exposure.cells) {
                ExposureMatch::Set { ev, clamped } => {
                    out.write_exposure = Some(ev);
                    // `+0.0`, never `-0.0`.
                    let shown = (ev * 10.0).round() / 10.0 + 0.0;
                    let clamp = if clamped { " (clamped)" } else { "" };
                    format!(
                        "Exposure set to {shown:+.1} EV from the room; refine it with the slider{clamp}"
                    )
                }
                ExposureMatch::NotInView => "The model is not in view".to_owned(),
            };
            info!("light: {note}");
            out.notes.push(note);
        }
        if i.exposure.pressed {
            out.consume_match = true;
            if out.can_match && !self.awaiting {
                self.awaiting = true;
                out.request_probe = true;
            }
        }
    }

    fn mode_of(&self, env: &EnvironmentSettings, setting: &str) -> LightingMode {
        let mut e = env.clone();
        e.lighting_mode = setting.to_owned();
        LightingMode::from_settings(&e, self.feature())
    }
}

/// The plugin, added by `plugin` with `XrPlugin.passthrough`.
pub(crate) struct LightPlugin;

impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Control>()
            .init_resource::<XrFeatures>()
            .init_resource::<HudData>()
            .init_resource::<Lighting>()
            .add_message::<AskPermissionAgain>()
            .add_systems(XrSessionCreated, new_session)
            .add_systems(XrPreDestroySession, end_session)
            .init_resource::<DepthReadback>()
            .add_systems(
                Update,
                (super::band::request, control)
                    .chain()
                    .after(crate::tracking::hands::write_feature),
            );
        // Dv-11: inert until the quest runner's entity names it (Q-308, Q-390).
        #[cfg(target_os = "android")]
        super::dv11::add(app);
    }
}

/// A new session: the camera may be found absent, and toasted, once more (Q-161).
fn new_session(mut control: ResMut<Control>) {
    control.session = true;
    control.unsupported = false;
    control.toasted = false;
}

/// The camera goes with its session.
fn end_session(mut control: ResMut<Control>) {
    control.stop();
    control.session = false;
}

/// An instance as auto local's band reads it: its pose, its bounds, whether it is selected.
type BandItem = (
    Entity,
    &'static GlobalTransform,
    &'static SplatBounds,
    Has<Selected>,
);

/// The left eye's pose and projection (Q-125: `XrCamera(0)`).
type Eye = (
    &'static XrCamera,
    &'static GlobalTransform,
    &'static Projection,
);

/// E14's grid over the layer.
const CELLS: bevy::math::UVec2 = bevy::math::UVec2::new(64, 48);

/// The frame's control: `Control::step` over the world's resources.
#[allow(clippy::too_many_arguments)]
fn control(
    mut control: ResMut<Control>,
    env: Option<Res<Environment>>,
    state: Option<Res<XrState>>,
    settings: Option<ResMut<SettingsRes>>,
    platform: Option<Res<PlatformRes>>,
    mut again: MessageReader<AskPermissionAgain>,
    mut denied: Option<ResMut<DeniedPermissions>>,
    mut features: ResMut<XrFeatures>,
    mut hud: ResMut<HudData>,
    mut lighting: ResMut<Lighting>,
    readback: Res<DepthReadback>,
    hands: Option<Res<XrHands>>,
    instances: Query<BandItem>,
    (eyes, mut probe): (Query<Eye>, Option<ResMut<LayerProbe>>),
    mut commands: Commands,
) {
    let Some(mut settings) = settings else {
        return;
    };
    // Q-388: d_m from the head (the views' midpoint) to the depth band's instance's posed B_m
    // centre: "per instance"'s target while one is set, else the selected one (Q-383).
    let target = band_instance(
        lighting.per_instance,
        instances.iter().map(|(e, at, b, s)| (e, s, (at, b))),
    );
    let band =
        readback
            .latest
            .as_ref()
            .zip(hands)
            .zip(target)
            .map(|((depth, hands), (at, bounds))| Band {
                depth,
                d_m: at
                    .transform_point(bounds.aabb.center())
                    .distance(hands.head.translation),
            });
    let asked_again = again.read().any(|a| a.0 == Permission::Camera);
    // E14's exposure match (Q-396): the probe's result taken as it arrives; the measured
    // cells, while "per instance" has a target, its posed B_m through the left view.
    let result = probe.as_mut().and_then(|p| p.result.take());
    let cells = lighting
        .per_instance
        .and_then(|t| instances.get(t).ok())
        .zip(eyes.iter().find(|(eye, ..)| eye.0 == 0))
        .map(|((_, at, bounds, _), (_, view, projection))| {
            let clip_from_model =
                projection.get_clip_from_view() * view.to_matrix().inverse() * at.to_matrix();
            box_cells(clip_from_model, bounds.aabb.min, bounds.aabb.max, CELLS)
        });
    let out = control.step(
        platform.as_deref().map(|p| &*p.0),
        &Inputs {
            env: &settings.0.environment,
            env_mode: env.map_or(EnvironmentMode::Vr, |e| e.mode),
            running: state.as_deref() == Some(&XrState::Running),
            asked_again,
            now: Instant::now(),
            band,
            exposure: Exposure {
                pressed: lighting.match_exposure,
                probe: result.as_ref(),
                cells,
            },
        },
    );
    // Only a write touches the settings (persist's debounced save follows a change).
    if let Some(m) = out.write_mode {
        settings.0.environment.lighting_mode = m.to_owned();
    }
    if let Some(ev) = out.write_exposure {
        settings.0.environment.lighting_manual.exposure_ev = ev;
    }
    if out.consume_match {
        lighting.match_exposure = false;
    }
    if out.request_probe
        && let Some(p) = probe.as_mut()
    {
        p.request = true;
    }
    if lighting.can_match != out.can_match {
        lighting.can_match = out.can_match;
    }
    for text in out.notes {
        commands.trigger(Toast {
            text,
            level: ToastLevel::Info,
        });
    }
    if let Some(r) = out.write_reference {
        info!("light: the reference captured {r:?}");
        settings.0.environment.lighting_reference = Some(r);
    }
    if let (Some(on), Some(d)) = (out.listed, denied.as_deref_mut()) {
        let has = d.0.contains(&Permission::Camera);
        if on && !has {
            d.0.push(Permission::Camera);
        } else if !on && has {
            d.0.retain(|p| *p != Permission::Camera);
        }
    }
    for text in out.toasts {
        commands.trigger(Toast {
            text,
            level: ToastLevel::Warning,
        });
    }
    if let Some(t) = out.transform
        && lighting.transform != t
    {
        lighting.transform = t;
    }
    if lighting.too_dark != out.too_dark {
        lighting.too_dark = out.too_dark;
    }
    // Q-395: the kept auto mode, for the Environment tab's "Ask again".
    if lighting.denied_auto != out.denied_auto {
        lighting.denied_auto = out.denied_auto;
    }
    if hud.light_gains != out.light_gains {
        hud.light_gains = out.light_gains;
    }
    if features.lighting != out.feature {
        features.lighting = out.feature;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ftgsxr_app::{AppPaths, CameraThumb, ModelRef, PermissionHandle, PickHandle, PickKind};

    use super::*;

    /// Q-385: the camera runs only with an auto mode, the permission, a running session
    /// and passthrough mode, each condition necessary.
    #[test]
    fn wanted_needs_auto_permission_running_and_passthrough() {
        let p = EnvironmentMode::Passthrough;
        for mode in [LightingMode::AutoGlobal, LightingMode::AutoLocal] {
            assert!(wanted(&mode, true, true, p));
            assert!(!wanted(&mode, false, true, p));
            assert!(!wanted(&mode, true, false, p));
            assert!(!wanted(&mode, true, true, EnvironmentMode::Vr));
        }
        assert!(!wanted(&LightingMode::Off, true, true, p));
        assert!(!wanted(
            &LightingMode::Manual(Default::default()),
            true,
            true,
            p
        ));
    }

    /// A camera whose thumbnail the test sets, counting its starts and stops; `start`
    /// returns the queued results, then `Ok`.
    #[derive(Default)]
    struct FakeCamera {
        thumb: Mutex<Option<CameraThumb>>,
        results: Mutex<Vec<Result<(), String>>>,
        starts: Mutex<u32>,
        stops: Mutex<u32>,
    }

    impl CameraFrames for FakeCamera {
        fn start(&self) -> Result<(), String> {
            *self.starts.lock().unwrap() += 1;
            let mut r = self.results.lock().unwrap();
            if r.is_empty() { Ok(()) } else { r.remove(0) }
        }
        fn latest(&self) -> Option<CameraThumb> {
            self.thumb.lock().unwrap().clone()
        }
        fn stop(&self) {
            *self.stops.lock().unwrap() += 1;
        }
    }

    /// A platform answering permissions with `grant`, counting requests, with the camera.
    struct Fake {
        paths: AppPaths,
        grant: Mutex<bool>,
        asked: Mutex<Vec<Permission>>,
        camera: Option<Arc<FakeCamera>>,
    }

    impl Platform for Fake {
        fn paths(&self) -> &AppPaths {
            &self.paths
        }
        fn pick(&self, _: PickKind) -> PickHandle {
            PickHandle(0)
        }
        fn poll_pick(&self, _: PickHandle) -> Poll<Result<Vec<ModelRef>, String>> {
            Poll::Pending
        }
        fn request_permission(&self, p: Permission) -> PermissionHandle {
            let mut a = self.asked.lock().unwrap();
            a.push(p);
            PermissionHandle(a.len() as u32)
        }
        fn poll_permission(&self, _: PermissionHandle) -> Poll<bool> {
            Poll::Ready(*self.grant.lock().unwrap())
        }
        fn document_provider(&self) -> Option<Arc<dyn ftgsxr_app::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
            self.camera.clone().map(|c| c as Arc<dyn CameraFrames>)
        }
    }

    fn fake(grant: bool, camera: bool) -> (Fake, Option<Arc<FakeCamera>>) {
        let root = std::path::PathBuf::from("/nonexistent");
        let cam = camera.then(|| Arc::new(FakeCamera::default()));
        (
            Fake {
                paths: AppPaths {
                    config: root.clone(),
                    cache: root.clone(),
                    models: root.clone(),
                    scenes: root.clone(),
                    screenshots: root.clone(),
                    logs: root,
                },
                grant: Mutex::new(grant),
                asked: Mutex::default(),
                camera: cam.clone(),
            },
            cam,
        )
    }

    fn env(mode: &str) -> EnvironmentSettings {
        let mut e = ftgsxr_io::Settings::default().environment;
        e.lighting_mode = mode.into();
        e
    }

    fn session() -> Control {
        Control {
            session: true,
            ..Control::default()
        }
    }

    /// A uniform thumbnail of `rgb` taken at `at_s`.
    fn thumb(rgb: [f32; 3], at_s: f64) -> CameraThumb {
        CameraThumb {
            width: 4,
            height: 2,
            rgb: rgb.repeat(8),
            at_s,
            camera: None,
        }
    }

    /// A frame at `now` in passthrough mode, running, applying the writes to `e`.
    fn frame(c: &mut Control, p: &Fake, e: &mut EnvironmentSettings, now: Instant) -> Outputs {
        frame_with(c, p, e, now, EnvironmentMode::Passthrough, false)
    }

    fn frame_with(
        c: &mut Control,
        p: &Fake,
        e: &mut EnvironmentSettings,
        now: Instant,
        env_mode: EnvironmentMode,
        asked_again: bool,
    ) -> Outputs {
        frame_exposure(c, p, e, now, env_mode, asked_again, Exposure::default())
    }

    fn frame_exposure(
        c: &mut Control,
        p: &Fake,
        e: &mut EnvironmentSettings,
        now: Instant,
        env_mode: EnvironmentMode,
        asked_again: bool,
        exposure: Exposure,
    ) -> Outputs {
        let out = c.step(
            Some(p),
            &Inputs {
                env: e,
                env_mode,
                running: true,
                asked_again,
                now,
                band: None,
                exposure,
            },
        );
        if let Some(ev) = out.write_exposure {
            e.lighting_manual.exposure_ev = ev;
        }
        if let Some(m) = out.write_mode {
            e.lighting_mode = m.into();
        }
        if let Some(r) = out.write_reference {
            e.lighting_reference = Some(r);
        }
        out
    }

    /// Q-384: without a session or without the platform's camera the feature is
    /// `Absent`, and nothing is asked.
    #[test]
    fn step_absent_without_session_or_camera() {
        let now = Instant::now();
        let (p, _) = fake(true, true);
        let mut e = env("auto-global");
        let mut c = Control::default();
        assert_eq!(frame(&mut c, &p, &mut e, now).feature, FeatureState::Absent);
        let (p, _) = fake(true, false);
        let mut c = session();
        assert_eq!(frame(&mut c, &p, &mut e, now).feature, FeatureState::Absent);
        assert!(p.asked.lock().unwrap().is_empty());
    }

    /// Q-384, Q-348: the permission is asked only when the camera would otherwise run:
    /// not in VR mode with an auto setting, then on the first passthrough frame.
    #[test]
    fn step_asks_the_camera_only_where_it_would_run() {
        let now = Instant::now();
        let (p, _) = fake(true, true);
        let mut e = env("auto-global");
        let mut c = session();
        frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Vr, false);
        assert!(p.asked.lock().unwrap().is_empty(), "asked in VR mode");
        frame(&mut c, &p, &mut e, now);
        assert_eq!(*p.asked.lock().unwrap(), [Permission::Camera]);
    }

    /// Q-348, Q-355, Q-261, Q-322: asked on the first want; a denial writes `manual`,
    /// lists the permission and toasts with Y "light estimation"; asked again on a
    /// re-selection and on `AskPermissionAgain`.
    #[test]
    fn step_permission_flow() {
        let now = Instant::now();
        let (p, _) = fake(false, true);
        let mut e = env("manual");
        let mut c = session();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(out.feature, FeatureState::Available);
        assert!(
            p.asked.lock().unwrap().is_empty(),
            "not asked in manual mode"
        );
        e.lighting_mode = "auto-global".into();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(*p.asked.lock().unwrap(), [Permission::Camera]);
        assert_eq!(out.feature, FeatureState::Denied);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(out.listed, Some(true));
        assert_eq!(
            out.toasts,
            ["E-XR-PERMISSION: Permission Camera denied; feature light estimation unavailable"]
        );
        // Nothing more while denied.
        let out = frame(&mut c, &p, &mut e, now);
        assert!(out.toasts.is_empty());
        assert_eq!(p.asked.lock().unwrap().len(), 1);
        // Selecting an auto mode again asks once more.
        e.lighting_mode = "auto-local".into();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(p.asked.lock().unwrap().len(), 2);
        assert_eq!(out.feature, FeatureState::Denied);
        assert_eq!(e.lighting_mode, "manual");
        // "Ask again", now granted.
        *p.grant.lock().unwrap() = true;
        let out = frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Passthrough, true);
        assert_eq!(p.asked.lock().unwrap().len(), 3);
        assert_eq!(out.listed, Some(false));
        assert_eq!(out.feature, FeatureState::Available);
        assert_eq!(c.flow, Flow::Granted);
    }

    /// Q-384, Q-395 (#691): a denial in auto local or auto global keeps that mode; the
    /// Environment tab's "Ask again" writes it back and sends `AskPermissionAgain`, which
    /// asks once; on a grant the effective mode is that auto mode with no other input and
    /// the kept mode ends.
    #[test]
    fn step_ask_again_restores_the_denied_auto_mode() {
        for (setting, mode) in [
            ("auto-local", LightingMode::AutoLocal),
            ("auto-global", LightingMode::AutoGlobal),
        ] {
            let now = Instant::now();
            let (p, cam) = fake(false, true);
            let cam = cam.unwrap();
            let mut e = env(setting);
            let mut c = session();
            let out = frame(&mut c, &p, &mut e, now);
            assert_eq!(out.feature, FeatureState::Denied, "{setting}");
            assert_eq!(e.lighting_mode, "manual", "{setting}");
            assert_eq!(out.denied_auto, Some(setting), "kept, {setting}");
            // Kept while denied.
            let out = frame(&mut c, &p, &mut e, now);
            assert_eq!(out.denied_auto, Some(setting), "{setting}");
            // "Ask again": the tab writes the kept mode and sends the message.
            *p.grant.lock().unwrap() = true;
            e.lighting_mode = out.denied_auto.unwrap().into();
            let out = frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Passthrough, true);
            assert_eq!(p.asked.lock().unwrap().len(), 2, "asked once, {setting}");
            assert_eq!(out.listed, Some(false), "{setting}");
            assert_eq!(out.feature, FeatureState::Available, "{setting}");
            assert_eq!(e.lighting_mode, setting, "{setting}");
            assert_eq!(c.mode_of(&e, &e.lighting_mode), mode, "{setting}");
            assert!(out.transform.is_some(), "the auto mode in force, {setting}");
            assert_eq!(*cam.starts.lock().unwrap(), 1, "the camera runs, {setting}");
            assert_eq!(out.denied_auto, None, "a grant ends it, {setting}");
            assert!(out.toasts.is_empty(), "{setting}");
        }
    }

    /// Q-384, Q-395, Q-261, Q-322 (#691): after "Ask again", a second denial writes
    /// `manual` again, toasts once and keeps the mode again; the Comfort tab's
    /// `AskPermissionAgain` asks without changing the setting or the kept mode.
    #[test]
    fn step_ask_again_denied_again_writes_manual() {
        let now = Instant::now();
        let (p, _) = fake(false, true);
        let mut e = env("auto-local");
        let mut c = session();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(out.toasts.len(), 1);
        // The Environment tab's "Ask again", denied again.
        e.lighting_mode = out.denied_auto.unwrap().into();
        let out = frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Passthrough, true);
        assert_eq!(p.asked.lock().unwrap().len(), 2, "asked once");
        assert_eq!(out.feature, FeatureState::Denied);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(out.listed, Some(true));
        assert_eq!(
            out.toasts,
            ["E-XR-PERMISSION: Permission Camera denied; feature light estimation unavailable"]
        );
        assert_eq!(out.denied_auto, Some("auto-local"));
        let out = frame(&mut c, &p, &mut e, now);
        assert!(out.toasts.is_empty(), "toasted once");
        // The Comfort tab's ask, denied: the setting and the kept mode stay.
        let out = frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Passthrough, true);
        assert_eq!(p.asked.lock().unwrap().len(), 3);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(out.denied_auto, Some("auto-local"));
        // The Comfort tab's ask, granted: the setting stays manual.
        *p.grant.lock().unwrap() = true;
        let out = frame_with(&mut c, &p, &mut e, now, EnvironmentMode::Passthrough, true);
        assert_eq!(p.asked.lock().unwrap().len(), 4);
        assert_eq!(out.feature, FeatureState::Available);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(out.denied_auto, None);
    }

    /// Q-392, Q-161: a start whose error begins with E-XR-UNSUPPORTED gives `Absent` for
    /// the session, writes `manual` and toasts once; another error leaves `Available` and
    /// retries 2 s later.
    #[test]
    fn step_start_errors() {
        let now = Instant::now();
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        cam.results.lock().unwrap().push(Err(
            "the passthrough camera could not be opened: code 3".into()
        ));
        let mut e = env("auto-global");
        let mut c = session();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(*cam.starts.lock().unwrap(), 1);
        assert_eq!(out.feature, FeatureState::Available);
        assert_eq!(e.lighting_mode, "auto-global");
        frame(&mut c, &p, &mut e, now + Duration::from_secs(1));
        assert_eq!(*cam.starts.lock().unwrap(), 1, "not before 2 s");
        cam.results
            .lock()
            .unwrap()
            .push(Err("E-XR-UNSUPPORTED: no passthrough camera".into()));
        let out = frame(&mut c, &p, &mut e, now + Duration::from_secs(2));
        assert_eq!(*cam.starts.lock().unwrap(), 2);
        assert_eq!(out.feature, FeatureState::Absent);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(
            out.toasts,
            ["E-XR-UNSUPPORTED: Not available on this runtime (passthrough camera)"]
        );
        assert!(out.transform.is_none(), "manual's transform is persist's");
        e.lighting_mode = "auto-global".into();
        let out = frame(&mut c, &p, &mut e, now + Duration::from_secs(5));
        assert_eq!(out.feature, FeatureState::Absent);
        assert_eq!(*cam.starts.lock().unwrap(), 2, "absent for the session");
    }

    /// Q-392, Q-384: a start whose error begins with E-XR-PERMISSION is a denial: `Denied`,
    /// `manual` written back, listed and toasted with Y "light estimation", not retried;
    /// "ask again", granted, starts the camera again.
    #[test]
    fn step_start_permission_error_is_a_denial() {
        let now = Instant::now();
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        cam.results.lock().unwrap().push(Err(
            "E-XR-PERMISSION: the camera permissions are not granted".into(),
        ));
        let mut e = env("auto-global");
        let mut c = session();
        let out = frame(&mut c, &p, &mut e, now);
        assert_eq!(*cam.starts.lock().unwrap(), 1);
        assert_eq!(out.feature, FeatureState::Denied);
        assert_eq!(c.flow, Flow::Denied);
        assert_eq!(e.lighting_mode, "manual");
        assert_eq!(out.listed, Some(true));
        assert_eq!(
            out.toasts,
            ["E-XR-PERMISSION: Permission Camera denied; feature light estimation unavailable"]
        );
        let out = frame(&mut c, &p, &mut e, now + Duration::from_secs(5));
        assert_eq!(*cam.starts.lock().unwrap(), 1, "not retried");
        assert!(out.toasts.is_empty());
        e.lighting_mode = "auto-global".into();
        let out = frame_with(
            &mut c,
            &p,
            &mut e,
            now + Duration::from_secs(6),
            EnvironmentMode::Passthrough,
            true,
        );
        assert_eq!(out.listed, Some(false));
        assert_eq!(out.feature, FeatureState::Available);
        assert_eq!(*cam.starts.lock().unwrap(), 2);
    }

    /// Q-385, Q-389: the smoothed estimate captured into `lighting_reference` when it is
    /// `None` and the warm-up is over; the gains into `Lighting.transform` and
    /// `HudData.light_gains`; the identity in VR mode with the camera stopped; the smoother
    /// and the warm-up restarted at a start.
    #[test]
    fn step_reference_and_gains() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        e.lighting_gamma = 1.0;
        let mut c = session();
        // Asked and granted, started; no thumbnail yet: the identity, the camera running.
        let out = frame(&mut c, &p, &mut e, at(0.0));
        assert_eq!(*cam.starts.lock().unwrap(), 1);
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        assert_eq!(out.light_gains, Some(Vec3::ONE));
        assert_eq!(e.lighting_reference, None);
        // Two thumbnails 1.5 s apart: captured as the reference, gains 1.
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 10.0));
        frame(&mut c, &p, &mut e, at(0.1));
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 11.5));
        let out = frame(&mut c, &p, &mut e, at(0.2));
        let r = e.lighting_reference.expect("captured");
        assert!((r[0] - 0.2).abs() < 1e-6 && (r[1] - 0.2).abs() < 1e-6);
        assert!(
            (out.transform.unwrap().gain - Vec3::ONE)
                .abs()
                .max_element()
                < 1e-6
        );
        // A doubled luma, long after (the smoother at the sample): g = 2 at γ = 1.
        *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 1000.0));
        let out = frame(&mut c, &p, &mut e, at(0.3));
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::splat(2.0)).abs().max_element() < 1e-5, "{g}");
        assert_eq!(out.light_gains, Some(g));
        assert_eq!(e.lighting_reference, Some(r), "captured once");
        // VR mode: the identity, the camera stopped, no HUD gains.
        let out = frame_with(&mut c, &p, &mut e, at(0.4), EnvironmentMode::Vr, false);
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        assert_eq!(out.light_gains, None);
        assert_eq!(*cam.stops.lock().unwrap(), 1);
        // Back in passthrough: a new start, the warm-up again, so the identity with the
        // saved reference until 1.5 s of `at_s`.
        *cam.thumb.lock().unwrap() = Some(thumb([0.8, 0.8, 0.8], 1000.01));
        let out = frame(&mut c, &p, &mut e, at(0.5));
        assert_eq!(*cam.starts.lock().unwrap(), 2);
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        assert_eq!(out.light_gains, Some(Vec3::ONE));
        *cam.thumb.lock().unwrap() = Some(thumb([0.8, 0.8, 0.8], 1001.0));
        let out = frame(&mut c, &p, &mut e, at(0.6));
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        // The smoother restarted too: after the warm-up the estimate is the fourfold
        // luma alone (g = 4), not averaged with the thumbnails before the stop.
        *cam.thumb.lock().unwrap() = Some(thumb([0.8, 0.8, 0.8], 1001.51));
        let out = frame(&mut c, &p, &mut e, at(0.7));
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::splat(4.0)).abs().max_element() < 1e-5, "{g}");
        assert_eq!(e.lighting_reference, Some(r), "the saved reference kept");
        // Manual mode: the transform is persist's; the camera stopped.
        e.lighting_mode = "manual".into();
        let out = frame(&mut c, &p, &mut e, at(0.8));
        assert!(out.transform.is_none(), "manual's transform is persist's");
        assert_eq!(out.light_gains, None);
    }

    /// Q-385's warm-up: no capture while one thumbnail has arrived since the start, however
    /// long the wall clock runs, nor before the newest thumbnail's `at_s` is 1.5 s (3τ)
    /// after the first one's; the identity meanwhile; a capture at the first thumbnail
    /// after both, with gains 1.
    #[test]
    fn step_captures_after_the_warm_up() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        let mut c = session();
        frame(&mut c, &p, &mut e, at(0.0));
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 10.0));
        for k in 1..19 {
            let out = frame(&mut c, &p, &mut e, at(0.1 * k as f64));
            assert_eq!(e.lighting_reference, None, "one thumbnail, frame {k}");
            assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
            assert!(!out.too_dark);
        }
        // The wall clock is the frames'; `at_s` is the shim's.
        for (k, at_s) in [10.5, 11.0, 11.49].into_iter().enumerate() {
            *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], at_s));
            let out = frame(&mut c, &p, &mut e, at(1.85 + 0.01 * k as f64));
            assert_eq!(e.lighting_reference, None, "at_s {at_s}");
            assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        }
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 11.5));
        let out = frame(&mut c, &p, &mut e, at(1.9));
        let r = e.lighting_reference.expect("captured after 1.5 s");
        for v in r {
            assert!((v - 0.2).abs() < 1e-6, "{r:?}");
        }
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::ONE).abs().max_element() < 1e-6, "{g}");
    }

    /// Q-385, Q-386: after the warm-up, no capture while Ȳ or a channel mean is below
    /// 1/255 (one code value), `too_dark` set and the identity meanwhile; a capture once
    /// the smoothed estimate rises above the floor, `too_dark` then clear.
    #[test]
    fn step_waits_above_the_floor() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        for dark in [[0.002, 0.002, 0.002], [0.5, 0.5, 0.003]] {
            let (p, cam) = fake(true, true);
            let cam = cam.unwrap();
            let mut e = env("auto-global");
            let mut c = session();
            frame(&mut c, &p, &mut e, at(0.0));
            *cam.thumb.lock().unwrap() = Some(thumb(dark, 10.0));
            let out = frame(&mut c, &p, &mut e, at(0.1));
            assert!(!out.too_dark, "{dark:?}: not during the warm-up");
            for (k, at_s) in [11.5, 11.6, 11.7].into_iter().enumerate() {
                *cam.thumb.lock().unwrap() = Some(thumb(dark, at_s));
                let out = frame(&mut c, &p, &mut e, at(0.2 + 0.1 * k as f64));
                assert_eq!(e.lighting_reference, None, "{dark:?} at {at_s}");
                assert!(out.too_dark, "{dark:?} at {at_s}");
                assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
            }
            *cam.thumb.lock().unwrap() = Some(thumb([0.5, 0.5, 0.5], 11.8));
            let out = frame(&mut c, &p, &mut e, at(0.6));
            let r = e.lighting_reference.expect("captured above the floor");
            assert!(r.iter().all(|&v| v >= 1.0 / 255.0), "{r:?}");
            assert!(!out.too_dark);
        }
    }

    /// Q-385: "re-capture reference" (the tab writing `None`) after the warm-up captures
    /// at the next new thumbnail, not at a frame without one.
    #[test]
    fn step_recapture_after_the_warm_up_takes_the_next_thumbnail() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        let mut c = session();
        frame(&mut c, &p, &mut e, at(0.0));
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 10.0));
        frame(&mut c, &p, &mut e, at(0.1));
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 11.5));
        frame(&mut c, &p, &mut e, at(0.2));
        assert!(e.lighting_reference.is_some(), "captured");
        e.lighting_reference = None;
        let out = frame(&mut c, &p, &mut e, at(0.3));
        assert_eq!(e.lighting_reference, None, "no new thumbnail");
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        assert!(!out.too_dark);
        *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 100.0));
        frame(&mut c, &p, &mut e, at(0.4));
        let r = e
            .lighting_reference
            .expect("captured at the next thumbnail");
        assert!((r[0] - 0.4).abs() < 1e-5, "{r:?}");
    }

    /// Q-385: a saved reference that passes the floor is kept, the transform the manual
    /// form (the identity at the default sliders) during the warm-up and E14's gains after
    /// it; a saved reference below the floor or not finite reads as `None` and is captured
    /// again after the warm-up.
    #[test]
    fn step_saved_reference_after_the_warm_up() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let run = |saved: [f32; 4]| {
            let (p, cam) = fake(true, true);
            let cam = cam.unwrap();
            let mut e = env("auto-global");
            e.lighting_gamma = 1.0;
            e.lighting_reference = Some(saved);
            let mut c = session();
            frame(&mut c, &p, &mut e, at(0.0));
            *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 10.0));
            let first = frame(&mut c, &p, &mut e, at(0.1));
            *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 11.0));
            let second = frame(&mut c, &p, &mut e, at(0.2));
            for out in [&first, &second] {
                assert_eq!(out.transform, Some(ColorTransform::IDENTITY), "{saved:?}");
                assert_eq!(out.light_gains, Some(Vec3::ONE));
            }
            assert_eq!(
                e.lighting_reference.map(|r| r.map(f32::to_bits)),
                Some(saved.map(f32::to_bits)),
                "kept during the warm-up"
            );
            *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 11.5));
            let out = frame(&mut c, &p, &mut e, at(0.3));
            (out, e.lighting_reference)
        };
        // Ȳ = 0.2 against an estimate of 0.4: g = 2 at γ = 1.
        let saved = [0.2; 4];
        let (out, kept) = run(saved);
        assert_eq!(kept, Some(saved));
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::splat(2.0)).abs().max_element() < 1e-5, "{g}");
        for saved in [
            [0.001, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.0],
            [0.2, f32::NAN, 0.2, 0.2],
        ] {
            let (out, captured) = run(saved);
            let r = captured.expect("captured");
            assert!(r.iter().all(|v| (v - 0.4).abs() < 1e-5), "{saved:?}: {r:?}");
            let g = out.transform.unwrap().gain;
            assert!((g - Vec3::ONE).abs().max_element() < 1e-5, "{saved:?}: {g}");
        }
    }

    /// Q-385: the floor takes Ȳ, r̄, ḡ and b̄ finite and at least 1/255.
    #[test]
    fn passes_floor_needs_four_finite_components_of_a_code_value() {
        let est = |luma, rgb| LightEstimate { luma, rgb };
        let f = 1.0 / 255.0;
        assert!(passes_floor(&est(f, [f; 3])));
        assert!(passes_floor(&est(0.5, [1.0, 0.2, 0.01])));
        assert!(!passes_floor(&est(0.0039, [0.5; 3])));
        assert!(!passes_floor(&est(0.5, [0.5, 0.0039, 0.5])));
        assert!(!passes_floor(&est(f32::NAN, [0.5; 3])));
        assert!(!passes_floor(&est(0.5, [0.5, 0.5, f32::INFINITY])));
    }

    /// U-20's clause (E14, revision 52) at U-20's tolerance 10⁻⁶, through the control,
    /// where the product is formed (nothing public composes it): manual EV 1 with auto
    /// g = √2 (a doubled luma at γ = 0.5) gives `Lighting.transform` g = 2√2, while
    /// `HudData.light_gains` stays E14's √2 (Q-394).
    #[test]
    fn step_manual_offset_composes_with_the_auto_gains() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        e.lighting_gamma = 0.5;
        e.lighting_manual.exposure_ev = 1.0;
        let mut c = session();
        frame(&mut c, &p, &mut e, at(0.0));
        // Two thumbnails 1.5 s apart: the warm-up over, the reference captured.
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 10.0));
        frame(&mut c, &p, &mut e, at(0.1));
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 11.5));
        frame(&mut c, &p, &mut e, at(0.2));
        assert!(e.lighting_reference.is_some(), "captured");
        *cam.thumb.lock().unwrap() = Some(thumb([0.4, 0.4, 0.4], 1000.0));
        let out = frame(&mut c, &p, &mut e, at(0.3));
        let g = out.transform.unwrap().gain;
        let want = 2.0 * 2f32.sqrt();
        assert!((g - Vec3::splat(want)).abs().max_element() <= 1e-6, "{g}");
        let hud = out.light_gains.unwrap();
        assert!(
            (hud - Vec3::splat(2f32.sqrt())).abs().max_element() <= 1e-6,
            "{hud}"
        );
    }

    /// E14 (revision 52): where the gains are not in force (started without an estimate,
    /// the warm-up, the capture waiting for the floor, VR mode) the transform is the
    /// manual form alone; with E14's gains at their clamp of 4, EV 1 gives 8 (the product
    /// is not clamped); EV −1 at the next frame halves E14's gains.
    #[test]
    fn step_manual_offset_alone_or_unclamped() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let close = |g: Vec3, want: f32| {
            assert!(
                (g - Vec3::splat(want)).abs().max_element() < 1e-5,
                "{g} vs {want}"
            );
        };
        // The capture waiting for the floor: the manual form alone, `too_dark` set.
        {
            let (p, cam) = fake(true, true);
            let cam = cam.unwrap();
            let mut e = env("auto-global");
            e.lighting_manual.exposure_ev = 1.0;
            let mut c = session();
            frame(&mut c, &p, &mut e, at(0.0));
            *cam.thumb.lock().unwrap() = Some(thumb([0.002; 3], 10.0));
            frame(&mut c, &p, &mut e, at(0.1));
            *cam.thumb.lock().unwrap() = Some(thumb([0.002; 3], 11.5));
            let out = frame(&mut c, &p, &mut e, at(0.2));
            assert!(out.too_dark);
            close(out.transform.unwrap().gain, 2.0);
            close(out.light_gains.unwrap(), 1.0);
        }
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        e.lighting_gamma = 1.0;
        e.lighting_manual.exposure_ev = 1.0;
        let mut c = session();
        // Started, no thumbnail yet: the manual form alone (2), E14's part 1.
        let out = frame(&mut c, &p, &mut e, at(0.0));
        close(out.transform.unwrap().gain, 2.0);
        close(out.light_gains.unwrap(), 1.0);
        // The warm-up: still the manual form alone.
        *cam.thumb.lock().unwrap() = Some(thumb([0.01, 0.01, 0.01], 10.0));
        let out = frame(&mut c, &p, &mut e, at(0.1));
        close(out.transform.unwrap().gain, 2.0);
        // The reference at Ȳ = 0.01, then Ȳ = 1 at γ = 1: E14's g clamped at 4.
        *cam.thumb.lock().unwrap() = Some(thumb([0.01, 0.01, 0.01], 11.5));
        frame(&mut c, &p, &mut e, at(0.2));
        *cam.thumb.lock().unwrap() = Some(thumb([1.0, 1.0, 1.0], 1000.0));
        let out = frame(&mut c, &p, &mut e, at(0.3));
        close(out.transform.unwrap().gain, 8.0);
        close(out.light_gains.unwrap(), 4.0);
        // EV −1 at the next frame: 0.5 × 4.
        e.lighting_manual.exposure_ev = -1.0;
        let out = frame(&mut c, &p, &mut e, at(0.4));
        close(out.transform.unwrap().gain, 2.0);
        // VR mode: the manual form alone.
        let out = frame_with(&mut c, &p, &mut e, at(0.5), EnvironmentMode::Vr, false);
        close(out.transform.unwrap().gain, 0.5);
        assert_eq!(out.light_gains, None);
    }

    /// Q-385: no new thumbnail for 2 s while started counts as a stop; the camera is
    /// started again while the conditions hold.
    #[test]
    fn step_stale_camera_restarts() {
        let t0 = Instant::now();
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        let mut c = session();
        frame(&mut c, &p, &mut e, t0);
        frame(&mut c, &p, &mut e, t0 + Duration::from_millis(1900));
        assert_eq!(*cam.stops.lock().unwrap(), 0);
        frame(&mut c, &p, &mut e, t0 + Duration::from_secs(2));
        assert_eq!(*cam.stops.lock().unwrap(), 1);
        assert_eq!(*cam.starts.lock().unwrap(), 2);
    }

    /// E14 local, Q-388, through the control: a thumbnail whose left half is (0.8, 0.4,
    /// 0.2) and right half (0.2, 0.2, 0.2), a fake depth with a wall at 2 m over its left
    /// half. Auto local with the band at d_m = 2 m takes the statistics over the band (the
    /// left half's colour); without the band (no provider, or no selection) and in auto
    /// global they are the frame's mean (0.5, 0.3, 0.2).
    #[test]
    fn step_auto_local_takes_the_band() {
        use super::super::band::DepthSnapshot;
        use ftgsxr_app::{Camera, Fov};
        let t0 = Instant::now();
        let fov = Fov::Angles {
            left: -0.7,
            right: 0.7,
            up: 0.6,
            down: -0.6,
        };
        let pose = (
            bevy::math::Vec3::new(0.0, 1.6, 0.0),
            bevy::math::Quat::IDENTITY,
        );
        let depth = DepthSnapshot {
            stored: (0..320 * 320)
                .map(|i| if i % 320 < 160 { 0.95 } else { 1.0 })
                .collect(),
            camera: Camera::from_stage_pose(pose.0, pose.1, fov, 320, 320),
            near: 0.1,
            far: f32::INFINITY,
        };
        let rgb: Vec<f32> = (0..64 * 48)
            .flat_map(|i| {
                if i % 64 < 32 {
                    [0.8, 0.4, 0.2]
                } else {
                    [0.2; 3]
                }
            })
            .collect();
        let shot = |at_s| CameraThumb {
            width: 64,
            height: 48,
            rgb: rgb.clone(),
            at_s,
            camera: Some(Camera::from_stage_pose(pose.0, pose.1, fov, 64, 48)),
        };
        let band = Band {
            depth: &depth,
            d_m: 2.0,
        };
        let cases = [
            ("auto-local", Some(band), [0.8, 0.4, 0.2]),
            ("auto-local", None, [0.5, 0.3, 0.2]),
            ("auto-global", Some(band), [0.5, 0.3, 0.2]),
        ];
        for (k, (mode, band, want)) in cases.into_iter().enumerate() {
            let (p, cam) = fake(true, true);
            let cam = cam.unwrap();
            let e = env(mode);
            let mut c = session();
            let mut step = |at: f64| {
                c.step(
                    Some(&p),
                    &Inputs {
                        env: &e,
                        env_mode: EnvironmentMode::Passthrough,
                        running: true,
                        asked_again: false,
                        now: t0 + Duration::from_secs_f64(at),
                        band,
                        exposure: Exposure::default(),
                    },
                )
            };
            step(0.0);
            *cam.thumb.lock().unwrap() = Some(shot(1.0 + k as f64));
            step(0.1);
            let raw = c.raw.expect("an estimate");
            for (got, want) in raw.rgb.iter().zip(want) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "{mode}, band {}: {:?}",
                    band.is_some(),
                    raw.rgb
                );
            }
        }
    }

    /// The probe's sums over a 64 × 48 layer, one pixel per cell, linear grey `c` at α = 1.
    fn sums(c: f32) -> LayerSums {
        LayerSums {
            cells: vec![[c, c, c, 1.0]; 64 * 48],
            width: 64,
            height: 48,
            layer: bevy::math::UVec2::new(64, 48),
        }
    }

    /// The sRGB decode of code 0.5, the room's L_room under a flat thumbnail of 0.5.
    const HALF_CODE_LINEAR: f32 = 0.214_041_14;

    /// Auto global past the warm-up with a flat thumbnail of code 0.5.
    fn warm_auto(
        c: &mut Control,
        p: &Fake,
        cam: &FakeCamera,
        e: &mut EnvironmentSettings,
        t0: Instant,
    ) {
        frame(c, p, e, t0);
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 10.0));
        frame(c, p, e, t0 + Duration::from_millis(10));
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 11.5));
        frame(c, p, e, t0 + Duration::from_millis(20));
    }

    fn press() -> Exposure<'static> {
        Exposure {
            pressed: true,
            ..Exposure::default()
        }
    }

    /// Q-396: `can_match` false before the warm-up is over, true after it, false outside
    /// an auto mode, in VR mode and after a camera stop.
    #[test]
    fn step_can_match_needs_the_camera_past_its_warm_up() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        let mut c = session();
        assert!(!frame(&mut c, &p, &mut e, at(0)).can_match, "no thumbnail");
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 10.0));
        assert!(
            !frame(&mut c, &p, &mut e, at(10)).can_match,
            "one thumbnail"
        );
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 11.0));
        assert!(!frame(&mut c, &p, &mut e, at(20)).can_match, "1 s of at_s");
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 11.5));
        assert!(frame(&mut c, &p, &mut e, at(30)).can_match, "warmed up");
        assert!(
            frame(&mut c, &p, &mut e, at(40)).can_match,
            "same thumbnail"
        );
        let out = frame_with(&mut c, &p, &mut e, at(50), EnvironmentMode::Vr, false);
        assert!(!out.can_match, "VR mode");
        e.lighting_mode = "manual".into();
        assert!(!frame(&mut c, &p, &mut e, at(60)).can_match, "manual");
        e.lighting_mode = "off".into();
        assert!(!frame(&mut c, &p, &mut e, at(70)).can_match, "off");
        // Back in auto global: a new start, the warm-up again; then the camera goes stale.
        e.lighting_mode = "auto-global".into();
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 20.0));
        assert!(!frame(&mut c, &p, &mut e, at(80)).can_match, "restarted");
        *cam.thumb.lock().unwrap() = Some(thumb([0.5; 3], 21.5));
        assert!(frame(&mut c, &p, &mut e, at(90)).can_match);
        let out = frame(&mut c, &p, &mut e, at(90) + STALE);
        assert!(!out.can_match, "the camera stopped");
    }

    /// Q-396: the press consumed once with one probe request; the result writes the
    /// exposure and toasts; temperature, tint and the reference stay.
    #[test]
    fn step_match_writes_the_exposure_once() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        e.lighting_manual.exposure_ev = 0.5;
        e.lighting_manual.temperature = 0.3;
        e.lighting_manual.tint = -0.2;
        let mut c = session();
        warm_auto(&mut c, &p, &cam, &mut e, t0);
        let reference = e.lighting_reference;
        assert!(reference.is_some());
        let out = frame_exposure(
            &mut c,
            &p,
            &mut e,
            at(100),
            EnvironmentMode::Passthrough,
            false,
            press(),
        );
        assert!(out.consume_match && out.request_probe);
        assert_eq!(out.write_exposure, None);
        // The tab's flag cleared: no second request while the result is awaited.
        let out = frame(&mut c, &p, &mut e, at(110));
        assert!(!out.consume_match && !out.request_probe);
        // The model at half the room's linear luma: +1 EV from 0.5.
        let model = sums(HALF_CODE_LINEAR / 2.0);
        let out = frame_exposure(
            &mut c,
            &p,
            &mut e,
            at(120),
            EnvironmentMode::Passthrough,
            false,
            Exposure {
                probe: Some(&model),
                ..Exposure::default()
            },
        );
        let ev = out.write_exposure.expect("written");
        assert!((ev - 1.5).abs() < 1e-5, "{ev}");
        assert_eq!(
            out.notes,
            ["Exposure set to +1.5 EV from the room; refine it with the slider"]
        );
        assert!(out.toasts.is_empty());
        assert_eq!(e.lighting_manual.temperature, 0.3);
        assert_eq!(e.lighting_manual.tint, -0.2);
        assert_eq!(e.lighting_reference, reference);
        assert_eq!(out.write_reference, None);
        // A second result, not asked for, writes nothing.
        let out = frame_exposure(
            &mut c,
            &p,
            &mut e,
            at(130),
            EnvironmentMode::Passthrough,
            false,
            Exposure {
                probe: Some(&model),
                ..Exposure::default()
            },
        );
        assert_eq!(out.write_exposure, None);
        assert!(out.notes.is_empty());
    }

    /// Q-396: " (clamped)" when E14's clamp was hit; `NotInView` writes nothing and
    /// toasts that the model is not in view; a press while matching is unavailable is
    /// consumed without a request.
    #[test]
    fn step_match_clamped_and_not_in_view() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let (p, cam) = fake(true, true);
        let cam = cam.unwrap();
        let mut e = env("auto-global");
        let mut c = session();
        let out = frame_exposure(
            &mut c,
            &p,
            &mut e,
            at(0),
            EnvironmentMode::Passthrough,
            false,
            press(),
        );
        assert!(out.consume_match && !out.request_probe, "not available yet");
        warm_auto(&mut c, &p, &cam, &mut e, at(1));
        let cases = [
            (
                sums(HALF_CODE_LINEAR / 64.0),
                Some(4.0),
                "Exposure set to +4.0 EV from the room; refine it with the slider (clamped)",
            ),
            (sums(0.0), None, "The model is not in view"),
        ];
        for (k, (model, want, note)) in cases.iter().enumerate() {
            let t = at(100 + 20 * k as u64);
            let before = e.lighting_manual.exposure_ev;
            let out = frame_exposure(
                &mut c,
                &p,
                &mut e,
                t,
                EnvironmentMode::Passthrough,
                false,
                press(),
            );
            assert!(out.request_probe);
            let out = frame_exposure(
                &mut c,
                &p,
                &mut e,
                t + Duration::from_millis(10),
                EnvironmentMode::Passthrough,
                false,
                Exposure {
                    probe: Some(model),
                    ..Exposure::default()
                },
            );
            assert_eq!(out.write_exposure, *want);
            assert_eq!(out.notes, [*note]);
            if want.is_none() {
                assert_eq!(e.lighting_manual.exposure_ev, before);
            }
        }
    }
}
