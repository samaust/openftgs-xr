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
//!   again or `AskPermissionAgain(Permission::Camera)` asks once more (Q-355).
//! - The camera's life and the estimate (Q-385): the camera runs while the effective
//!   lighting mode is auto, the permission is granted, the session runs and the effective
//!   environment mode is passthrough; the smoother restarts at each start; no new
//!   thumbnail for 2 s while started counts as a stop. Each new thumbnail is pushed into
//!   the smoother; while `lighting_reference` is `None` the next smoothed estimate is
//!   written into it; the gains go into `Lighting.transform` in the auto modes (the
//!   identity in VR mode) and `HudData.light_gains` while the camera runs (Q-389).

use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use bevy::app::{App, Plugin, Update};
use bevy::ecs::message::MessageReader;
use bevy::log::{info, warn};
use bevy::math::Vec3;
use bevy::prelude::{
    Commands, GlobalTransform, IntoScheduleConfigs, Query, Res, ResMut, Resource, With,
};
use bevy_mod_xr::session::{XrPreDestroySession, XrSessionCreated, XrState};
use ftgsxr_app::{
    AskPermissionAgain, CameraFrames, DeniedPermissions, FeatureState, HudData, Lighting,
    Permission, Platform, PlatformRes, Selected, SettingsRes, Toast, ToastLevel, XrFeatures,
};
use ftgsxr_io::EnvironmentSettings;
use ftgsxr_render::ColorTransform;
use ftgsxr_render::SplatBounds;

use super::band::{DepthReadback, DepthSnapshot, thumb_depths};
use super::{LightEstimate, LightSmoother, LightingMode, band_mask, grey_world_gains, light_gains};
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
    /// `Lighting.transform`, in an effective auto mode only (Q-383).
    pub(crate) transform: Option<ColorTransform>,
    /// `HudData.light_gains` (Q-389).
    pub(crate) light_gains: Option<Vec3>,
    /// `lighting_mode` written back (Q-355, Q-392).
    pub(crate) write_mode: Option<&'static str>,
    /// `lighting_reference` captured (Q-385).
    pub(crate) write_reference: Option<[f32; 4]>,
    /// Toasts raised.
    pub(crate) toasts: Vec<String>,
    /// `Permission::Camera` listed in (true) or out of (false) `DeniedPermissions`.
    pub(crate) listed: Option<bool>,
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
    /// The camera is started: since when a thumbnail was last new.
    started: Option<Instant>,
    /// A failed start is retried from then on (Q-392).
    retry_after: Option<Instant>,
    /// The last thumbnail's `at_s`.
    pub(crate) last_at_s: Option<f64>,
    pub(crate) smoother: LightSmoother,
    /// The last new thumbnail's raw estimate (Dv-11).
    pub(crate) raw: Option<LightEstimate>,
    /// The new thumbnails since the plugin started (Dv-11).
    pub(crate) thumbs: u64,
    /// A thumbnail carried the camera (Dv-11).
    pub(crate) pose_seen: bool,
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
            started: None,
            retry_after: None,
            last_at_s: None,
            smoother: LightSmoother::new(TAU_S),
            raw: None,
            thumbs: 0,
            pose_seen: false,
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
    /// listed, toasted, and an auto setting written back to `manual`.
    fn deny(&mut self, out: &mut Outputs, setting: &mut String, auto: bool) {
        self.flow = Flow::Denied;
        out.listed = Some(true);
        if auto {
            *setting = "manual".into();
            out.write_mode = Some("manual");
        }
        let text = denied_text(Permission::Camera, FEATURE);
        warn!("{text}");
        out.toasts.push(text);
    }

    /// One frame (Q-384, Q-385, Q-389, Q-392).
    // Q-161, Q-261, Q-322, Q-348, Q-355, Q-384, Q-385, Q-389, Q-392
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
                    self.last_at_s = None;
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
            if let Some(e) = LightEstimate::from_thumb(&thumb, mask.as_deref()) {
                self.raw = Some(e);
                self.smoother.push(e, thumb.at_s);
                new = true;
            }
        }

        let auto = mode.is_auto();
        let mut reference = i.env.lighting_reference;
        if auto
            && reference.is_none()
            && new
            && let Some(e) = self.smoother.current()
        {
            let r = [e.luma, e.rgb[0], e.rgb[1], e.rgb[2]];
            reference = Some(r);
            out.write_reference = Some(r);
        }
        if auto {
            let t = match (i.env_mode, self.smoother.current()) {
                (EnvironmentMode::Vr, _) | (_, None) => ColorTransform::IDENTITY,
                (_, Some(e)) => match reference {
                    Some([luma, r, g, b]) => light_gains(
                        &e,
                        &LightEstimate {
                            luma,
                            rgb: [r, g, b],
                        },
                        i.env.lighting_gamma,
                    ),
                    None => grey_world_gains(&e),
                },
            };
            out.transform = Some(t);
        }
        out.light_gains = self
            .started
            .and_then(|_| out.transform.map(|t| t.gain).or(Some(Vec3::ONE)));
        self.last_auto = auto_setting(&setting);
        out.feature = self.feature();
        out
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
    selected: Query<(&GlobalTransform, &SplatBounds), With<Selected>>,
    mut commands: Commands,
) {
    let Some(mut settings) = settings else {
        return;
    };
    // Q-388: d_m from the head (the views' midpoint) to the selected instance's posed
    // B_m centre.
    let band = readback
        .latest
        .as_ref()
        .zip(hands)
        .zip(selected.single().ok())
        .map(|((depth, hands), (at, bounds))| Band {
            depth,
            d_m: at
                .transform_point(bounds.aabb.center())
                .distance(hands.head.translation),
        });
    let asked_again = again.read().any(|a| a.0 == Permission::Camera);
    let out = control.step(
        platform.as_deref().map(|p| &*p.0),
        &Inputs {
            env: &settings.0.environment,
            env_mode: env.map_or(EnvironmentMode::Vr, |e| e.mode),
            running: state.as_deref() == Some(&XrState::Running),
            asked_again,
            now: Instant::now(),
            band,
        },
    );
    // Only a write touches the settings (persist's debounced save follows a change).
    if let Some(m) = out.write_mode {
        settings.0.environment.lighting_mode = m.to_owned();
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
        let out = c.step(
            Some(p),
            &Inputs {
                env: e,
                env_mode,
                running: true,
                asked_again,
                now,
                band: None,
            },
        );
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

    /// Q-385, Q-389: the first smoothed estimate captured into `lighting_reference` when
    /// it is `None`; the gains into `Lighting.transform` and `HudData.light_gains`; the
    /// identity in VR mode with the camera stopped; the smoother restarted at a start.
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
        // The first thumbnail: captured as the reference, gains 1.
        *cam.thumb.lock().unwrap() = Some(thumb([0.2, 0.2, 0.2], 10.0));
        let out = frame(&mut c, &p, &mut e, at(0.1));
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
        let out = frame(&mut c, &p, &mut e, at(0.2));
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::splat(2.0)).abs().max_element() < 1e-5, "{g}");
        assert_eq!(out.light_gains, Some(g));
        assert_eq!(e.lighting_reference, Some(r), "captured once");
        // VR mode: the identity, the camera stopped, no HUD gains.
        let out = frame_with(&mut c, &p, &mut e, at(0.3), EnvironmentMode::Vr, false);
        assert_eq!(out.transform, Some(ColorTransform::IDENTITY));
        assert_eq!(out.light_gains, None);
        assert_eq!(*cam.stops.lock().unwrap(), 1);
        // Back in passthrough: a new start with the smoother restarted, so a thumbnail
        // of a fourfold luma is taken as it is (g = 4), not averaged with the old ones.
        *cam.thumb.lock().unwrap() = Some(thumb([0.8, 0.8, 0.8], 1000.01));
        let out = frame(&mut c, &p, &mut e, at(0.4));
        assert_eq!(*cam.starts.lock().unwrap(), 2);
        let g = out.transform.unwrap().gain;
        assert!((g - Vec3::splat(4.0)).abs().max_element() < 1e-5, "{g}");
        // Manual mode: the transform is persist's; the camera stopped.
        e.lighting_mode = "manual".into();
        let out = frame(&mut c, &p, &mut e, at(0.6));
        assert!(out.transform.is_none(), "manual's transform is persist's");
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
}
