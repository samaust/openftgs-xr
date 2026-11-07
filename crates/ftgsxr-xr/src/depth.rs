// SPDX-License-Identifier: Apache-2.0

//! The environment-depth provider (§1.9 Depth occlusion, E13, D-16, §2.1
//! `EnvironmentDepth`).
//!
//! - The control (main world): the provider runs while the effective mode is
//!   passthrough, `[environment] occlusion` is on and the running tier's `occlusion` flag
//!   is true (Q-338), once `Permission::Scene` is granted (Q-348, asked on the first
//!   turn-on; a denial turns the setting off with Q-261's toast, and turning it on again
//!   asks again, Q-355). It writes `XrFeatures.occlusion` (Q-336), `HudData.depth_image`
//!   (Q-345) and the main world's `SplatOcclusion` from the previous frame (Q-339), and
//!   creates the two views' `DepthFrame.image`s once the swapchain's size is known.
//! - The drive (render world, after the frame's begin and Bevy's prepared assets, before
//!   the camera passes): the adapter (`meta`) started and stopped as asked, hand removal
//!   at the start and on change (D-16), `acquire` called once per view (Q-349), the frame's
//!   `SplatOcclusion` written for the draw node (Q-339), empty while stopped (Q-338).
//!
//! Built with `XrPlugin.passthrough` (the Depth API needs passthrough, E.2); where the
//! runtime lacks XR\_META\_environment\_depth the feature is `Absent` and E-XR-UNSUPPORTED
//! is logged (Q-134, Q-161: the default is no explicit request, so no toast).

// Dv-7's device side (Q-346, Q-308), built for the device and the host tests.
#[cfg(any(test, target_os = "android"))]
mod dv7;
mod meta;

use std::sync::{Arc, Mutex};
use std::task::Poll;

use bevy::app::{App, Plugin, PostUpdate};
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::ecs::message::MessageReader;
use bevy::image::Image;
use bevy::log::{info, warn};
use bevy::prelude::{
    Commands, DetectChanges, GlobalTransform, IntoScheduleConfigs, Res, ResMut, Resource,
    Transform, resource_exists,
};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureUsages};
use bevy::render::texture::GpuImage;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use bevy_mod_openxr::init::should_run_frame_loop;
use bevy_mod_openxr::resources::{OxrFrameState, Pipelined};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_openxr::spaces::OxrSpaceExt;
use bevy_mod_xr::session::{XrPreDestroySession, XrRootTransform, XrSessionCreated};
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::{
    AskPermissionAgain, DeniedPermissions, FeatureState, HudData, Permission, PlatformRes,
    SettingsRes, Toast, ToastLevel, XrFeatures,
};
use ftgsxr_core::{Camera, Fov, UVec2};
use ftgsxr_render::{DepthFrame, SplatGpu, SplatOcclusion, SplatTierState};

use crate::error::{XrError, log_warning};
use crate::passthrough::{Environment, EnvironmentMode};
use crate::tracking::hands::{Flow, denied_text};

use meta::{FrameCtx, MetaDepth};

/// The extension the provider needs (Appendix E.1).
pub(crate) const EXTENSION: &str = "XR_META_environment_depth";

/// The environment-depth port (§2.1, E13): the adapter over XR\_META\_environment\_depth
/// on the headset, none where the runtime lacks it. `start` starts the provider with the
/// hand-removal option (D-16), `stop` stops it. `acquire` is called once per view per
/// frame (Q-349): the first call of a frame acquires the runtime's image and returns the
/// left view's frame, the second the right view's from the same image, further calls
/// `None` until the next frame.
// Q-349
pub trait EnvironmentDepth: Send + Sync {
    /// Starts the provider, with hand removal on or off (D-16).
    fn start(&mut self, hand_removal: bool) -> Result<(), XrError>;
    /// The next view's depth frame of this frame's image, `None` once both were given or
    /// before the first image.
    fn acquire(&mut self) -> Option<DepthFrame>;
    /// Stops the provider.
    fn stop(&mut self);
}

/// What one runtime acquire gave (Q-349).
pub(crate) enum Acquired {
    /// A new image: its left and right views' frames, aged 0.
    New(Box<[DepthFrame; 2]>),
    /// No image newer than the last (none yet, or the same capture): the last frames,
    /// whose copied images still hold it, one frame older.
    Same,
    /// The acquire failed: no frames.
    Failed,
}

/// The per-frame sequence of `EnvironmentDepth::acquire` (Q-349): the first call of a
/// frame acquires once and gives the left view, the second the right view of the same
/// image, the rest `None` until `next_frame`. `age_frames` counts the frames since the
/// last new image (Q-339).
// Q-339, Q-349
#[derive(Default)]
pub(crate) struct Views {
    given: usize,
    frames: Option<[DepthFrame; 2]>,
}

impl Views {
    /// A new frame: the next call acquires again.
    pub(crate) fn next_frame(&mut self) {
        self.given = 0;
    }

    /// Forgets the frames (the provider stopped).
    pub(crate) fn clear(&mut self) {
        *self = Views::default();
    }

    /// The next view's frame; `acquire` is the runtime's one acquire of the frame.
    pub(crate) fn next(&mut self, acquire: impl FnOnce() -> Acquired) -> Option<DepthFrame> {
        let i = self.given;
        if i >= 2 {
            return None;
        }
        if i == 0 {
            match acquire() {
                Acquired::New(frames) => self.frames = Some(*frames),
                Acquired::Same => {
                    for f in self.frames.iter_mut().flatten() {
                        f.age_frames = f.age_frames.saturating_add(1);
                    }
                }
                Acquired::Failed => self.frames = None,
            }
        }
        self.given += 1;
        self.frames.as_ref().map(|f| f[i].clone())
    }
}

/// Q-349: this frame's frames of `depth`, left then right: `acquire` called once per
/// view.
// Q-349
pub(crate) fn frames_of(depth: &mut dyn EnvironmentDepth) -> Vec<DepthFrame> {
    (0..2).map_while(|_| depth.acquire()).collect()
}

/// What the render world's drive needs beyond §2.1's port (as `anchors`' driver):
/// the swapchain's size and length once created, hand removal on change (D-16), the
/// frame's context, the running state and the failed calls (Dv-7).
pub(crate) trait DepthDriver: EnvironmentDepth {
    /// The swapchain's size and image count, once created (the first start).
    fn swapchain(&self) -> Option<(UVec2, usize)>;
    /// Hand removal on or off while the provider exists (D-16).
    fn set_hand_removal(&mut self, on: bool) -> Result<(), XrError>;
    /// A new frame: its context; the next `acquire` acquires again (Q-349).
    fn begin_frame(&mut self, ctx: FrameCtx);
    /// Whether the provider is started.
    fn running(&self) -> bool;
    /// The runtime calls that failed in this session.
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    fn failures(&self) -> u32;
}

/// Q-338: whether the provider runs, from the effective mode, `[environment] occlusion`
/// and the running tier's `occlusion` flag: all three must allow it.
// Q-338
pub(crate) fn wanted(mode: EnvironmentMode, setting: bool, tier: bool) -> bool {
    mode == EnvironmentMode::Passthrough && setting && tier
}

/// Q-341: a depth view's camera in the stage space. `view` is the view's pose in the
/// primary reference space the acquire named; it is composed with the tracking root's
/// global transform `root`, as the eyes' views are (bevy\_mod\_openxr's render-world
/// view update, source-log row 361), and built with E6's `Fov::Angles` at the image's
/// `size`.
// Q-341
pub(crate) fn depth_camera(
    view: Transform,
    fov: Fov,
    size: UVec2,
    root: &GlobalTransform,
) -> Camera {
    let (_, rotation, translation) = root.mul_transform(view).to_scale_rotation_translation();
    Camera::from_stage_pose(translation, rotation, fov, size.x, size.y)
}

/// What the main world asks of the render world's drive, extracted each frame.
#[derive(Resource, Clone, Debug, Default, PartialEq, ExtractResource)]
pub(crate) struct DepthRequest {
    /// Run the provider (Q-338, and the permission granted, Q-348).
    pub(crate) run: bool,
    /// `[environment] hand_removal` (D-16).
    pub(crate) hand_removal: bool,
    /// `[environment] occlusion_soft` and `occlusion_delta_m` (E13).
    pub(crate) soft: bool,
    pub(crate) delta_m: f32,
    /// The two views' `DepthFrame.image`s, once created.
    pub(crate) images: Option<[Handle<Image>; 2]>,
}

/// What the render world tells the main world, a frame late (Q-339).
#[derive(Clone, Debug, Default)]
pub(crate) struct Report {
    /// The swapchain's size and image count, once created.
    pub(crate) swapchain: Option<(UVec2, usize)>,
    /// The runtime refused the provider as unsupported (the system lacks depth).
    pub(crate) unsupported: bool,
    /// The last frame's occlusion.
    pub(crate) frames: Vec<DepthFrame>,
}

/// `Report`, shared by the two worlds.
#[derive(Resource, Clone, Default)]
pub(crate) struct Shared(pub(crate) Arc<Mutex<Report>>);

impl Shared {
    fn get(&self) -> Report {
        self.0.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

/// The main world's control state: the permission flow (kept across sessions, as the
/// hands' and anchors'), the setting last seen (Q-355), the images, whether the
/// instance enabled the extension and the session's warning.
#[derive(Resource, Default)]
pub(crate) struct Control {
    pub(crate) flow: Flow,
    last_setting: bool,
    images: Option<[Handle<Image>; 2]>,
    images_size: Option<UVec2>,
    extension: bool,
    /// A session exists (set at its creation, cleared before its destruction).
    session: bool,
    warned: bool,
}

/// The render world's provider (PRD #565: a crate-private render-world resource) and
/// the drive's record (Dv-7 reads it).
#[derive(Resource, Default)]
pub(crate) struct Provider {
    pub(crate) depth: Option<Box<dyn DepthDriver>>,
    /// A start that failed is not retried until the request goes off.
    failed: bool,
    hand_removal: Option<bool>,
}

/// The splat device in the render world, for the import (Q-339).
#[derive(Resource, Default)]
struct RenderGpu(Option<SplatGpu>);

/// The depth plugin, added by `plugin` with `XrPlugin.passthrough`.
pub(crate) struct DepthPlugin;

impl Plugin for DepthPlugin {
    fn build(&self, app: &mut App) {
        let shared = Shared::default();
        app.init_resource::<Control>()
            .init_resource::<DepthRequest>()
            .init_resource::<XrFeatures>()
            .init_resource::<HudData>()
            .add_message::<AskPermissionAgain>()
            .insert_resource(shared.clone())
            .add_plugins(ExtractResourcePlugin::<DepthRequest>::default())
            .add_systems(XrSessionCreated, new_session)
            .add_systems(XrPreDestroySession, end_session)
            .add_systems(PostUpdate, control);
        // Q-134: E-XR-UNSUPPORTED as a warning where the instance lacks the extension.
        if let Some(enabled) = app.world().get_resource::<OxrEnabledExtensions>() {
            let on = enabled.0.raw().meta_environment_depth;
            if !on {
                log_warning(&XrError::Unsupported(EXTENSION));
            }
            app.world_mut().resource_mut::<Control>().extension = on;
        }
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(shared)
                .init_resource::<Provider>()
                .init_resource::<RenderGpu>()
                .add_systems(ExtractSchedule, extract_gpu)
                .add_systems(XrPreDestroySession, destroy)
                .add_systems(
                    Render,
                    drive
                        .in_set(RenderSystems::PrepareResources)
                        .run_if(resource_exists::<OxrSession>)
                        .run_if(should_run_frame_loop),
                );
        }
        // Dv-7: inert until the quest runner's entity names it (Q-308, Q-346).
        #[cfg(target_os = "android")]
        dv7::add(app);
    }
}

/// A new session may warn of an unsupported system once more (Q-134).
fn new_session(mut control: ResMut<Control>) {
    control.warned = false;
    control.session = true;
}

/// Without a session the feature is absent.
fn end_session(mut control: ResMut<Control>) {
    control.session = false;
}

/// The splat device into the render world, for the import.
fn extract_gpu(gpu: Extract<Option<Res<SplatGpu>>>, mut ours: ResMut<RenderGpu>) {
    let now = gpu.as_deref().cloned();
    if now.is_some() != ours.0.is_some() || gpu.as_ref().is_some_and(|g| g.is_changed()) {
        ours.0 = now;
    }
}

/// Asks for `Permission::Scene` through the platform; without one (a test app without
/// `AppPlugin`) it counts as granted, as `LinuxPlatform` grants every one (Q-276).
// Q-348
fn ask(platform: Option<&PlatformRes>) -> Flow {
    match platform {
        Some(p) => Flow::Asking(p.0.request_permission(Permission::Scene)),
        None => Flow::Granted,
    }
}

/// The main world's frame: the run condition (Q-338), the permission (Q-348, Q-355),
/// the request, the images once the swapchain's size is known (Q-339), the previous
/// frame's occlusion into the main world's `SplatOcclusion` and `HudData.depth_image`
/// (Q-345), `XrFeatures.occlusion` (Q-336).
// Q-261, Q-336, Q-338, Q-339, Q-345, Q-348, Q-355
#[allow(clippy::too_many_arguments)]
fn control(
    mut control: ResMut<Control>,
    mut request: ResMut<DepthRequest>,
    shared: Res<Shared>,
    env: Option<Res<Environment>>,
    settings: Option<ResMut<SettingsRes>>,
    tier: Option<Res<SplatTierState>>,
    platform: Option<Res<PlatformRes>>,
    mut again: MessageReader<AskPermissionAgain>,
    mut denied: Option<ResMut<DeniedPermissions>>,
    mut features: ResMut<XrFeatures>,
    mut hud: ResMut<HudData>,
    images: Option<ResMut<Assets<Image>>>,
    occlusion: Option<ResMut<SplatOcclusion>>,
    mut commands: Commands,
) {
    let report = shared.get();
    let available = control.extension && control.session && !report.unsupported;
    if report.unsupported && !control.warned {
        control.warned = true;
        log_warning(&XrError::Unsupported(EXTENSION));
    }
    let mut settings = settings;
    let (setting, hand_removal, soft, delta_m) = settings.as_deref().map_or_else(
        || {
            let e = ftgsxr_io::Settings::default().environment;
            (
                e.occlusion,
                e.hand_removal,
                e.occlusion_soft,
                e.occlusion_delta_m,
            )
        },
        |s| {
            let e = &s.0.environment;
            (
                e.occlusion,
                e.hand_removal,
                e.occlusion_soft,
                e.occlusion_delta_m,
            )
        },
    );
    let mode = env.map_or(EnvironmentMode::Vr, |e| e.mode);
    let tier_on = tier.is_some_and(|t| t.tier.occlusion);
    let want = available && wanted(mode, setting, tier_on);

    // Q-348, Q-355: asked on the first turn-on; while denied, turning the setting on
    // again (the Environment tab's "Ask again" toggles it) or the Comfort tab's "ask
    // again" asks once more.
    let mut listed = |on: bool| {
        if let Some(d) = denied.as_deref_mut() {
            let has = d.0.contains(&Permission::Scene);
            if on && !has {
                d.0.push(Permission::Scene);
            } else if !on && has {
                d.0.retain(|p| *p != Permission::Scene);
            }
        }
    };
    let asked_again = again.read().any(|a| a.0 == Permission::Scene);
    let turned_on = setting && !control.last_setting;
    if control.flow == Flow::Denied && (asked_again || turned_on) {
        control.flow = ask(platform.as_deref());
        listed(false);
    }
    if want && control.flow == Flow::Idle {
        control.flow = ask(platform.as_deref());
    }
    let mut setting = setting;
    if let Flow::Asking(h) = control.flow
        && let Some(platform) = platform.as_deref()
    {
        match platform.0.poll_permission(h) {
            Poll::Pending => {}
            Poll::Ready(true) => control.flow = Flow::Granted,
            Poll::Ready(false) => {
                control.flow = Flow::Denied;
                listed(true);
                // Q-355: the feature is off, and so is its switch.
                if let Some(s) = settings.as_deref_mut() {
                    s.0.environment.occlusion = false;
                }
                setting = false;
                // Q-261
                let text = denied_text(Permission::Scene, "occlusion");
                warn!("{text}");
                commands.trigger(Toast {
                    text,
                    level: ToastLevel::Warning,
                });
            }
        }
    }
    control.last_setting = setting;

    // Q-339: the two views' images, once the swapchain's size is known, again when a
    // new session's swapchain has another size.
    if let (Some((size, _)), Some(mut images)) = (report.swapchain, images)
        && control.images_size != Some(size)
    {
        if let Some(old) = control.images.take() {
            for h in old {
                images.remove(&h);
            }
        }
        let mut make = || {
            let mut image = Image::new_uninit(
                Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                meta::FORMAT,
                RenderAssetUsages::default(),
            );
            image.texture_descriptor.usage =
                TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
            images.add(image)
        };
        control.images = Some([make(), make()]);
        control.images_size = Some(size);
        info!(
            "depth: the views' images created at {} x {}",
            size.x, size.y
        );
    }

    let want_request = DepthRequest {
        run: want && setting && control.flow == Flow::Granted,
        hand_removal,
        soft,
        delta_m,
        images: control.images.clone(),
    };
    if *request != want_request {
        *request = want_request;
    }

    // Q-339: the main world's occlusion is the previous frame's.
    let frames = report.frames;
    let depth_image = report
        .swapchain
        .zip(frames.first())
        .map(|((size, _), f)| (size, f.age_frames));
    if hud.depth_image != depth_image {
        hud.depth_image = depth_image;
    }
    let main = SplatOcclusion {
        frames,
        soft,
        delta_m,
    };
    match occlusion {
        Some(mut o) => *o = main,
        None => commands.insert_resource(main),
    }

    // Q-336
    let state = if !available {
        FeatureState::Absent
    } else if control.flow == Flow::Denied {
        FeatureState::Denied
    } else {
        FeatureState::Available
    };
    if features.occlusion != state {
        features.occlusion = state;
    }
}

/// The render world's frame (Q-338, Q-339, Q-341, Q-349, D-16): the adapter made once
/// per session, started and stopped as asked, hand removal on change, two acquires, the
/// frame's `SplatOcclusion` (empty while stopped), the report for the main world.
// Q-338, Q-339, Q-341, Q-349
#[allow(clippy::too_many_arguments)]
fn drive(
    request: Res<DepthRequest>,
    session: Res<OxrSession>,
    mut provider: ResMut<Provider>,
    shared: Res<Shared>,
    gpu: Res<RenderGpu>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    frame: Option<Res<OxrFrameState>>,
    pipelined: Option<Res<Pipelined>>,
    space: Option<Res<XrPrimaryReferenceSpace>>,
    root: Option<Res<XrRootTransform>>,
    occlusion: Option<ResMut<SplatOcclusion>>,
    mut commands: Commands,
) {
    let provider = &mut *provider;
    if provider.depth.is_none() && session.instance().exts().meta_environment_depth.is_some() {
        let session: &openxr::Session<openxr::AnyGraphics> = &session;
        provider.depth = Some(Box::new(MetaDepth::new(session.clone())));
    }
    let mut unsupported = false;
    let frames = match provider.depth.as_deref_mut() {
        Some(depth) => {
            step(
                depth,
                &request,
                &mut provider.failed,
                &mut provider.hand_removal,
                &mut unsupported,
            );
            match (depth.running(), frame, space) {
                (true, Some(frame), Some(space)) => {
                    // The eyes' rule (bevy_mod_openxr's `locate_views`, source-log row 361).
                    let time = if pipelined.is_some() {
                        openxr::Time::from_nanos(
                            frame.predicted_display_time.as_nanos()
                                + frame.predicted_display_period.as_nanos(),
                        )
                    } else {
                        frame.predicted_display_time
                    };
                    let targets = request.images.as_ref().and_then(|[l, r]| {
                        Some([
                            (l.clone(), gpu_images.get(l)?.texture.clone()),
                            (r.clone(), gpu_images.get(r)?.texture.clone()),
                        ])
                    });
                    depth.begin_frame(FrameCtx {
                        time,
                        space: space.0.0.as_raw_openxr_space(),
                        root: root.map_or(GlobalTransform::IDENTITY, |r| r.0),
                        gpu: gpu.0.clone(),
                        targets,
                    });
                    frames_of(depth)
                }
                _ => Vec::new(),
            }
        }
        None => Vec::new(),
    };
    if let Ok(mut r) = shared.0.lock() {
        r.swapchain = provider.depth.as_deref().and_then(|d| d.swapchain());
        r.unsupported |= unsupported;
        r.frames = frames.clone();
    }
    let now = SplatOcclusion {
        frames,
        soft: request.soft,
        delta_m: request.delta_m,
    };
    match occlusion {
        Some(mut o) => *o = now,
        None => commands.insert_resource(now),
    }
}

/// Starts or stops `depth` as `request` asks, hand removal at the start and on change
/// (D-16); a failed start waits for the request to go off; an unsupported system is
/// reported.
// D-16, Q-338, Q-357
pub(crate) fn step(
    depth: &mut dyn DepthDriver,
    request: &DepthRequest,
    failed: &mut bool,
    hand_removal: &mut Option<bool>,
    unsupported: &mut bool,
) {
    if !request.run {
        *failed = false;
        if depth.running() {
            depth.stop();
        }
        return;
    }
    if !depth.running() && !*failed {
        match depth.start(request.hand_removal) {
            Ok(()) => *hand_removal = Some(request.hand_removal),
            Err(e) => {
                *failed = true;
                *unsupported |= matches!(e, XrError::Unsupported(_));
                log_warning(&e);
            }
        }
    } else if depth.running() && *hand_removal != Some(request.hand_removal) {
        if let Err(e) = depth.set_hand_removal(request.hand_removal) {
            warn!("depth: {e}");
        }
        *hand_removal = Some(request.hand_removal);
    }
}

/// Before the session is destroyed: the provider stopped and destroyed with its
/// swapchain (the adapter's `Drop`), the report and the render world's frames cleared.
fn destroy(
    mut provider: ResMut<Provider>,
    shared: Res<Shared>,
    occlusion: Option<ResMut<SplatOcclusion>>,
) {
    *provider = Provider::default();
    // The draw node binds nothing of the old session (Q-338: empty while stopped).
    if let Some(mut o) = occlusion {
        o.frames.clear();
    }
    if let Ok(mut r) = shared.0.lock() {
        *r = Report::default();
    }
}

#[cfg(test)]
mod tests {
    use bevy::app::App;
    use bevy::prelude::Vec3 as BVec3;
    use ftgsxr_core::{DVec3, Quat, Vec3};

    use super::*;

    /// A frame whose near plane names it.
    fn frame(near: f32) -> DepthFrame {
        DepthFrame {
            image: bevy::asset::Handle::default(),
            camera: Camera::from_stage_pose(
                Vec3::ZERO,
                Quat::IDENTITY,
                Fov::Vertical { radians: 1.0 },
                4,
                4,
            ),
            near,
            far: 10.0,
            age_frames: 0,
        }
    }

    /// A provider over `Views` whose runtime is a script of outcomes, counting its
    /// acquires.
    struct Fake {
        views: Views,
        script: Vec<Acquired>,
        acquires: usize,
    }

    impl EnvironmentDepth for Fake {
        fn start(&mut self, _hand_removal: bool) -> Result<(), XrError> {
            Ok(())
        }
        fn acquire(&mut self) -> Option<DepthFrame> {
            let (script, acquires) = (&mut self.script, &mut self.acquires);
            self.views.next(|| {
                *acquires += 1;
                if script.is_empty() {
                    Acquired::Same
                } else {
                    script.remove(0)
                }
            })
        }
        fn stop(&mut self) {
            self.views.clear();
        }
    }

    fn nears_ages(frames: &[DepthFrame]) -> Vec<(f32, u32)> {
        frames.iter().map(|f| (f.near, f.age_frames)).collect()
    }

    /// Q-349: one runtime acquire per frame however often `acquire` is called; the left
    /// then the right view of the same image, then `None`; nothing before the first
    /// image; a frame without a newer image gives the last two, a frame older; a failed
    /// acquire gives none.
    #[test]
    fn views_acquire_once_per_frame_and_give_left_then_right() {
        let mut fake = Fake {
            views: Views::default(),
            script: vec![
                Acquired::Same,
                Acquired::New(Box::new([frame(0.1), frame(0.2)])),
                Acquired::Same,
                Acquired::Failed,
            ],
            acquires: 0,
        };
        // Frame 1: no image yet.
        fake.views.next_frame();
        assert!(frames_of(&mut fake).is_empty());
        assert_eq!(fake.acquires, 1);
        // Frame 2: a new image, left then right, then nothing more this frame.
        fake.views.next_frame();
        assert_eq!(nears_ages(&frames_of(&mut fake)), vec![(0.1, 0), (0.2, 0)]);
        assert!(fake.acquire().is_none());
        assert_eq!(fake.acquires, 2);
        // Frame 3: no newer image: the same two, a frame older.
        fake.views.next_frame();
        assert_eq!(nears_ages(&frames_of(&mut fake)), vec![(0.1, 1), (0.2, 1)]);
        assert_eq!(fake.acquires, 3);
        // Frame 4: the acquire fails: no frames.
        fake.views.next_frame();
        assert!(frames_of(&mut fake).is_empty());
        assert_eq!(fake.acquires, 4);
    }

    /// A driver that records its calls; `start` fails while `refuse` holds an error.
    #[derive(Default)]
    struct Driver {
        running: bool,
        starts: Vec<bool>,
        removals: Vec<bool>,
        stops: usize,
        refuse: Option<XrError>,
    }

    impl EnvironmentDepth for Driver {
        fn start(&mut self, hand_removal: bool) -> Result<(), XrError> {
            self.starts.push(hand_removal);
            match self.refuse.take() {
                Some(e) => Err(e),
                None => {
                    self.running = true;
                    Ok(())
                }
            }
        }
        fn acquire(&mut self) -> Option<DepthFrame> {
            None
        }
        fn stop(&mut self) {
            self.stops += 1;
            self.running = false;
        }
    }

    impl DepthDriver for Driver {
        fn swapchain(&self) -> Option<(UVec2, usize)> {
            None
        }
        fn set_hand_removal(&mut self, on: bool) -> Result<(), XrError> {
            self.removals.push(on);
            Ok(())
        }
        fn begin_frame(&mut self, _ctx: FrameCtx) {}
        fn running(&self) -> bool {
            self.running
        }
        fn failures(&self) -> u32 {
            0
        }
    }

    /// Q-338, D-16, Q-357: started with hand removal when the request runs, hand removal
    /// sent again only on change, stopped when the request stops; a refused start is not
    /// retried until the request goes off, and an unsupported system is reported.
    #[test]
    fn step_starts_stops_and_sends_hand_removal_on_change() {
        let mut d = Driver::default();
        let (mut failed, mut removal, mut unsupported) = (false, None, false);
        let mut req = DepthRequest {
            run: true,
            ..DepthRequest::default()
        };
        let mut go = |d: &mut Driver, req: &DepthRequest| {
            step(d, req, &mut failed, &mut removal, &mut unsupported);
        };
        go(&mut d, &req);
        go(&mut d, &req);
        assert_eq!(d.starts, vec![false]);
        assert!(d.removals.is_empty(), "the start carried it");
        req.hand_removal = true;
        go(&mut d, &req);
        go(&mut d, &req);
        assert_eq!(d.removals, vec![true]);
        req.run = false;
        go(&mut d, &req);
        go(&mut d, &req);
        assert_eq!(d.stops, 1);

        d.refuse = Some(XrError::Unsupported(EXTENSION));
        req.run = true;
        go(&mut d, &req);
        go(&mut d, &req);
        assert_eq!(d.starts, vec![false, true], "not retried");
        req.run = false;
        go(&mut d, &req);
        req.run = true;
        go(&mut d, &req);
        assert_eq!(
            d.starts,
            vec![false, true, true],
            "retried after the request went off"
        );
        assert!(d.running);
        drop(go);
        assert!(unsupported);
    }

    /// A platform that answers every permission with `grant`.
    struct Answering(ftgsxr_io::AppPaths, Arc<Mutex<bool>>);

    impl ftgsxr_app::Platform for Answering {
        fn paths(&self) -> &ftgsxr_io::AppPaths {
            &self.0
        }
        fn pick(&self, _: ftgsxr_app::PickKind) -> ftgsxr_app::PickHandle {
            ftgsxr_app::PickHandle(0)
        }
        fn poll_pick(
            &self,
            _: ftgsxr_app::PickHandle,
        ) -> Poll<Result<Vec<ftgsxr_io::ModelRef>, String>> {
            Poll::Ready(Ok(Vec::new()))
        }
        fn request_permission(&self, _: Permission) -> ftgsxr_app::PermissionHandle {
            ftgsxr_app::PermissionHandle(0)
        }
        fn poll_permission(&self, _: ftgsxr_app::PermissionHandle) -> Poll<bool> {
            Poll::Ready(*self.1.lock().expect("unpoisoned"))
        }
        fn document_provider(&self) -> Option<Arc<dyn ftgsxr_io::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<Arc<dyn ftgsxr_app::CameraFrames>> {
            None
        }
    }

    /// Q-348, Q-261, Q-355, Q-336: in passthrough with the setting and the tier on, the
    /// first turn-on asks; a denial toasts naming "occlusion", lists the permission, marks
    /// the feature `Denied`, turns the setting off and does not run; turning the setting on
    /// again asks again, and a grant runs the provider.
    #[test]
    fn control_asks_on_turn_on_and_again_after_a_denial() {
        let grant = Arc::new(Mutex::new(false));
        let dir = std::path::PathBuf::from("/nonexistent");
        let mut app = App::new();
        app.insert_resource(Control {
            extension: true,
            session: true,
            ..Control::default()
        })
        .init_resource::<DepthRequest>()
        .init_resource::<Shared>()
        .init_resource::<XrFeatures>()
        .init_resource::<HudData>()
        .init_resource::<DeniedPermissions>()
        .add_message::<AskPermissionAgain>()
        .insert_resource(Environment {
            mode: EnvironmentMode::Passthrough,
            passthrough_available: true,
            style: crate::passthrough::PassthroughStyle::default(),
        })
        .insert_resource(SettingsRes(ftgsxr_io::Settings::default()))
        .insert_resource(SplatTierState::of(&ftgsxr_render::QualityTier {
            occlusion: true,
            ..ftgsxr_render::QualityTier::conformance()
        }))
        .insert_resource(PlatformRes(Arc::new(Answering(
            ftgsxr_io::AppPaths {
                config: dir.clone(),
                cache: dir.clone(),
                models: dir.clone(),
                scenes: dir.clone(),
                screenshots: dir.clone(),
                logs: dir,
            },
            grant.clone(),
        ))))
        .add_systems(bevy::app::Update, control);
        let toasts = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = toasts.clone();
        app.add_observer(move |t: bevy::prelude::On<Toast>| {
            seen.lock().expect("unpoisoned").push(t.text.clone());
        });
        for _ in 0..3 {
            app.update();
        }
        let w = app.world();
        assert!(!w.resource::<DepthRequest>().run);
        assert_eq!(w.resource::<XrFeatures>().occlusion, FeatureState::Denied);
        assert_eq!(w.resource::<DeniedPermissions>().0, vec![Permission::Scene]);
        assert!(!w.resource::<SettingsRes>().0.environment.occlusion);
        {
            let toasts = toasts.lock().expect("unpoisoned");
            assert_eq!(toasts.len(), 1, "{toasts:?}");
            assert!(
                toasts[0].contains("feature occlusion unavailable"),
                "{toasts:?}"
            );
        }

        *grant.lock().expect("unpoisoned") = true;
        app.world_mut()
            .resource_mut::<SettingsRes>()
            .0
            .environment
            .occlusion = true;
        for _ in 0..3 {
            app.update();
        }
        let w = app.world();
        assert!(w.resource::<DepthRequest>().run);
        assert_eq!(
            w.resource::<XrFeatures>().occlusion,
            FeatureState::Available
        );
        assert!(w.resource::<DeniedPermissions>().0.is_empty());
        assert_eq!(toasts.lock().expect("unpoisoned").len(), 1);
    }

    /// The pixel a world point projects to (E6: x = K (R_w p + t_w) / z).
    fn project(c: &Camera, p: Vec3) -> (f64, f64) {
        let q = c.world_to_camera.transform_point3(p.as_dvec3());
        let k = c.k * (q / q.z);
        (k.x, k.y)
    }

    /// Within 10⁻⁴ px: tan(π/4) in f32 is 1 − 6·10⁻⁸.
    fn close(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4
    }

    /// Q-341: a view 1.5 m up in the reference space, under a tracking root translated
    /// to (1, 0, 2): the camera sits at (1, 1.5, 2), a point 1 m ahead (−z) projects to the
    /// image's center and one 1 m ahead and 1 m right to its right edge (±45°, 100 px:
    /// f_x = 100 / 2 = 50). With the root also turned 90° about +y, ahead is −x.
    #[test]
    fn depth_camera_composes_the_view_with_a_translated_root() {
        let fov = Fov::Angles {
            left: -std::f32::consts::FRAC_PI_4,
            right: std::f32::consts::FRAC_PI_4,
            up: std::f32::consts::FRAC_PI_4,
            down: -std::f32::consts::FRAC_PI_4,
        };
        let size = UVec2::new(100, 100);
        let view = Transform::from_translation(BVec3::new(0.0, 1.5, 0.0));
        let root = GlobalTransform::from(Transform::from_translation(BVec3::new(1.0, 0.0, 2.0)));
        let c = depth_camera(view, fov, size, &root);
        assert!(
            (c.center() - DVec3::new(1.0, 1.5, 2.0)).length() < 1e-6,
            "{:?}",
            c.center()
        );
        assert!(close(project(&c, Vec3::new(1.0, 1.5, 1.0)), (50.0, 50.0)));
        assert!(close(project(&c, Vec3::new(2.0, 1.5, 1.0)), (100.0, 50.0)));

        let turned = GlobalTransform::from(
            Transform::from_translation(BVec3::new(1.0, 0.0, 2.0))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        );
        let c = depth_camera(view, fov, size, &turned);
        assert!((c.center() - DVec3::new(1.0, 1.5, 2.0)).length() < 1e-6);
        assert!(close(project(&c, Vec3::new(0.0, 1.5, 2.0)), (50.0, 50.0)));
    }

    /// Q-338: the provider runs only in passthrough mode with the setting on and the
    /// tier's flag true; each of the three alone turns it off.
    #[test]
    fn wanted_needs_passthrough_the_setting_and_the_tier() {
        use EnvironmentMode::{Passthrough, Vr};
        assert!(wanted(Passthrough, true, true));
        for (mode, setting, tier) in [
            (Vr, true, true),
            (Passthrough, false, true),
            (Passthrough, true, false),
            (Vr, false, true),
            (Vr, true, false),
            (Passthrough, false, false),
            (Vr, false, false),
        ] {
            assert!(!wanted(mode, setting, tier), "{mode:?} {setting} {tier}");
        }
    }
}
