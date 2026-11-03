// SPDX-License-Identifier: Apache-2.0

//! PRD #76's added test, the Bevy adapter renders like the headless renderer (source of
//! truth: R-1's criterion, §3.2): a Bevy `App` with `SplatPlugin { tier: conformance }`,
//! one `SplatCamera` with `OutputSpace::FileEncoded` rendering into an image target and
//! one `SplatModel` of P1 at the request's first item, read back, meets E15's tier
//! (PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %) against the reference render. It catches a broken
//! plugin wiring (loads, uploads, extraction, the two render nodes, the camera's
//! conversion to E6's camera); window and swapchain behaviour are M-4's I-n. The other
//! tests here check `LinearOver`'s resolve onto an sRGB target (Q-94), that
//! `FileEncoded` is not drawn onto one (Q-100, #178), and the GPU times of a frame that
//! skips stages 1–4 (Q-81, Q-115, #184), that a change of τ_m, the camera or an
//! instance's transform runs them again (#187) and a background change alone does not
//! (#201), and the components' requirements and D-6 settings (#202, #203).

use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, Last, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::visibility::Visibility;
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, ClearColorConfig, PerspectiveProjection, Projection,
    RenderTarget,
};
use bevy::color::Color;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::{Mat3, Quat, Vec3};
use bevy::mesh::MeshPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;
use bevy::render::view::Msaa;
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_core::{Camera as FtgsCamera, DMat3, RenderImage};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    FrameStats, OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad, SplatMode,
    SplatModel, SplatOverlays, SplatPlugin, SplatStats, SplatTime, SplatTransform,
};
use ftgsxr_testkit::{fixtures_dir, layer_messages, metrics, probe, reference_render};

/// E15's tier.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;
/// Frames the app may take to load, upload and draw before the test gives up.
const MAX_FRAMES: usize = 600;
/// Frames rendered after the upload before the read back is taken, so the read back
/// holds a frame that drew the model.
const SETTLE_FRAMES: usize = 4;

/// Inserted into the App's main world by the render error handler: Bevy reported a
/// validation, out-of-memory or internal error, or lost the device (§1.13: any message
/// fails a `gpu` test). Held per App, so a test sees only its own App's errors (#206,
/// Q-105).
#[derive(Resource)]
struct RenderErrored;

/// Whether `app`'s render error handler has fired.
fn render_errored(app: &App) -> bool {
    app.world().contains_resource::<RenderErrored>()
}

#[derive(Resource, Clone, Default)]
struct Seen {
    events: Arc<Mutex<Vec<SplatEvent>>>,
    readback: Arc<Mutex<Option<Vec<u8>>>>,
    stats: Arc<Mutex<Vec<FrameStats>>>,
}

/// The Bevy view transform of openftgs's camera: the center −R_wᵀ t_w, and the view axes
/// x = the camera's x, y = −(its y), z = −(its z) (Bevy looks along −z with y up).
fn bevy_transform(camera: &FtgsCamera) -> Transform {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    let center = -(r.transpose() * t);
    let flip = DMat3::from_diagonal(ftgsxr_core::DVec3::new(1.0, -1.0, -1.0));
    let world_from_view = (r.transpose() * flip).as_mat3();
    Transform {
        translation: center.as_vec3(),
        rotation: Quat::from_mat3(&Mat3::from_cols(
            world_from_view.x_axis,
            world_from_view.y_axis,
            world_from_view.z_axis,
        )),
        scale: Vec3::ONE,
    }
}

/// Bevy's perspective projection of a centered pinhole with square pixels, at the
/// engine's near plane 0.01 m (§1.7, Q-87).
fn bevy_projection(camera: &FtgsCamera) -> Projection {
    let k = camera.k;
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    assert_eq!(k.x_axis.x, k.y_axis.y, "square pixels");
    assert_eq!(
        (k.z_axis.x, k.z_axis.y),
        (w / 2.0, h / 2.0),
        "a centered principal point"
    );
    Projection::Perspective(PerspectiveProjection {
        fov: (2.0 * (h / 2.0 / k.y_axis.y).atan()) as f32,
        aspect_ratio: (w / h) as f32,
        near: 0.01,
        ..PerspectiveProjection::default()
    })
}

/// IEEE binary16 to f32.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The read back premultiplied file-space layer (C, A), rows of `w` 16-bit float texels
/// (padded to `row_bytes`), composited over `background` as a headless render does:
/// rgb = C + (1 − A)·c_bg, alpha = A (E7, §1.7).
fn composite(bytes: &[u8], w: u32, h: u32, background: [f32; 3]) -> RenderImage {
    let row_bytes = bytes.len() / h as usize;
    assert!(
        row_bytes >= 8 * w as usize,
        "{} bytes for {w} × {h}",
        bytes.len()
    );
    let mut rgb = Vec::new();
    let mut alpha = Vec::new();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let texel = &bytes[y * row_bytes + 8 * x..][..8];
            let v = |i: usize| half(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            let a = v(3);
            for (c, bg) in background.iter().enumerate() {
                rgb.push(v(c) + (1.0 - a) * bg);
            }
            alpha.push(a);
        }
    }
    RenderImage {
        width: w,
        height: h,
        rgb,
        alpha,
    }
}

/// #291: a scene dropped while a pipeline compiles would let the process exit with a
/// worker in the driver; not while panicking, so a failed test reports.
impl Drop for P1Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

/// #291 (§2.4 Threads): on bevy's `multi_threaded` pools (unified in by the workspace's
/// build) `PipelineCache` compiles on the async compute pool, so a frame may draw before
/// the splat or tonemapping pipelines exist, and a test returning while one compiles lets
/// the process exit with a worker inside the Vulkan driver. Runs frames until no render
/// pipeline is queued or compiling, at most 10 s of wall time; whether none is.
fn settle_pipelines(app: &mut App) -> bool {
    use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
    let busy = |app: &App| {
        app.sub_app(RenderApp)
            .world()
            .resource::<PipelineCache>()
            .pipelines()
            .filter(|p| {
                matches!(
                    p.state,
                    CachedPipelineState::Queued | CachedPipelineState::Creating(_)
                )
            })
            .count()
    };
    let start = std::time::Instant::now();
    while busy(app) > 0 {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            return false;
        }
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}

/// P1's first item drawn by a Bevy `App` with `SplatPlugin { tier: conformance }`: the
/// model loaded through `SplatLoad` at the item's timestamp, one `SplatCamera` writing
/// `output` into an image target of `format`.
struct P1Scene {
    app: App,
    seen: Seen,
    camera: Entity,
    model: Entity,
}

impl P1Scene {
    /// The scene, run until the model uploaded and a few frames more, so the next frames
    /// draw it.
    fn settled(output: OutputSpace, format: TextureFormat) -> P1Scene {
        let (_, request) = probe("P1");
        let item = &request.items[0];
        let (w, h) = (item.camera.width, item.camera.height);
        let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).expect("P1's file");

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            AssetPlugin::default(),
            WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..WindowPlugin::default()
            },
            RenderPlugin::default(),
            ImagePlugin::default(),
            MeshPlugin,
            CameraPlugin,
            CorePipelinePlugin,
            SplatPlugin {
                tier: QualityTier::conformance(),
            },
        ))
        .insert_resource(RenderErrorHandler(|error, main_world, _| {
            eprintln!("render error: {:?}: {}", error.ty, error.description);
            main_world.insert_resource(RenderErrored);
            RenderErrorPolicy::StopRendering
        }));
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();

        let seen = Seen::default();
        app.insert_resource(seen.clone());
        app.add_observer(|event: On<SplatEvent>, seen: Res<Seen>| {
            seen.events.lock().unwrap().push(event.event().clone());
        });
        // Every statistics block as it arrives, not only the last (#201, #187).
        app.add_systems(Last, |stats: Res<SplatStats>, seen: Res<Seen>| {
            if stats.is_changed()
                && let Some(s) = stats.0
            {
                seen.stats.lock().unwrap().push(s);
            }
        });

        let mut image = Image::new_target_texture(w, h, format, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);

        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera::default(),
                RenderTarget::Image(target.clone().into()),
                bevy_projection(&item.camera),
                bevy_transform(&item.camera),
                SplatCamera {
                    sort: SortMode::PerView,
                    output,
                },
            ))
            .id();
        let model = app
            .world_mut()
            .spawn((
                SplatLoad {
                    bytes: ModelBytes::Owned(bytes),
                    kind_hint: None,
                    reduce_to: None,
                },
                SplatTime {
                    tau: item.timestamp,
                    extrapolating: false,
                },
                Transform::IDENTITY,
            ))
            .id();
        app.world_mut().spawn(Readback::texture(target)).observe(
            |event: On<ReadbackComplete>, seen: Res<Seen>| {
                *seen.readback.lock().unwrap() = Some(event.data.clone());
            },
        );

        let mut scene = P1Scene {
            app,
            seen,
            camera,
            model,
        };
        let mut uploaded_at = None;
        for frame in 0..MAX_FRAMES {
            scene.frames(1);
            let events = scene.seen.events.lock().unwrap().clone();
            if let Some(SplatEvent::Failed { error, .. }) = events
                .iter()
                .find(|e| matches!(e, SplatEvent::Failed { .. }))
            {
                panic!("the load failed: {error}");
            }
            if uploaded_at.is_none()
                && events
                    .iter()
                    .any(|e| matches!(e, SplatEvent::Uploaded { entity, .. } if *entity == model))
            {
                uploaded_at = Some(frame);
            }
            if uploaded_at.is_some_and(|f| frame >= f + SETTLE_FRAMES) {
                break;
            }
        }
        assert!(
            uploaded_at.is_some(),
            "P1 uploaded within {MAX_FRAMES} frames"
        );
        assert!(
            settle_pipelines(&mut scene.app),
            "the render pipelines compiled within 10 s"
        );
        scene
    }

    /// Runs `n` frames, failing on any render error.
    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
            assert!(!render_errored(&self.app), "Bevy reported a render error");
        }
    }

    /// A fresh read back of the target, from a frame rendered after this call: two
    /// frames, then `more`.
    fn readback(&mut self, more: usize) -> Vec<u8> {
        *self.seen.readback.lock().unwrap() = None;
        self.frames(2 + more);
        self.seen
            .readback
            .lock()
            .unwrap()
            .clone()
            .expect("a read back of the image target")
    }

    /// The statistics blocks that arrived since the last call.
    fn take_stats(&self) -> Vec<FrameStats> {
        std::mem::take(&mut *self.seen.stats.lock().unwrap())
    }

    /// Whether the device has timestamp queries.
    fn timestamps(&self) -> bool {
        self.app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .features()
            .contains(WgpuFeatures::TIMESTAMP_QUERY)
    }
}

/// The target's bytes as read back a few frames after the upload, the `SplatStats`
/// then, and whether the device has timestamp queries; `unchanged_frames` more frames
/// are rendered before the read back is taken.
fn render_p1(
    output: OutputSpace,
    format: TextureFormat,
    unchanged_frames: usize,
) -> (Vec<u8>, SplatStats, bool) {
    let mut scene = P1Scene::settled(output, format);
    let data = scene.readback(unchanged_frames);
    let stats = *scene.app.world().resource::<SplatStats>();
    (data, stats, scene.timestamps())
}

/// The first item's reference render of P1.
fn p1_reference() -> (RenderImage, [f32; 3]) {
    let (_, request) = probe("P1");
    let id = request.items[0].id.clone();
    let reference = reference_render("P1")
        .into_iter()
        .find(|(i, _)| *i == id)
        .expect("the first item's reference")
        .1;
    (reference, request.background)
}

/// The statistics arrived two frames late, counted P1's primitives drawn, no overflow.
fn assert_stats(stats: SplatStats) {
    let s = stats.0.expect("statistics two frames late (§1.4 stage 8)");
    assert!(s.visible_splats > 0 && !s.overflow, "{s:?}");
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_renders_p1_like_the_headless_renderer() {
    let _layer = layer_messages(); // R-16, Q-105
    // The image target is 16-bit float RGBA, so the view's main texture is the layer's
    // format and `FileEncoded` holds the layer unrounded.
    let (data, stats, _) = render_p1(OutputSpace::FileEncoded, TextureFormat::Rgba16Float, 0);
    let (reference, background) = p1_reference();
    let ours = composite(&data, reference.width, reference.height, background);
    let m = metrics(&reference, &ours);
    eprintln!(
        "P1 through the Bevy adapter (FileEncoded): PSNR {:.2} dB, s {:.4}, s_alpha {:.4}",
        m.psnr, m.share, m.alpha
    );
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
        "E15's tier: {m:?}"
    );
    // A blank image would pass against a blank reference: P1's first item covers pixels.
    assert!(ours.alpha.iter().any(|&a| a > 0.1), "the model was drawn");
    assert_stats(stats);
}

/// `LinearOver` with a background (VR mode, Q-94): the layer goes through the resolve onto
/// an sRGB target, whose stored bytes are then the file's values of E7's
/// C + (1 − A)·c_bg (D-6, R-11), within 8-bit rounding: E15's color criteria against the
/// reference composited over the same background. The target is opaque, so alpha is not
/// compared.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_resolves_p1_over_a_background_onto_an_srgb_target() {
    let _layer = layer_messages(); // R-16, Q-105
    let background = [0.2, 0.4, 0.6];
    let (data, stats, _) = render_p1(
        OutputSpace::LinearOver {
            background: Some(background),
        },
        TextureFormat::Rgba8UnormSrgb,
        0,
    );
    assert_resolved_over(&data, background);
    assert_stats(stats);
}

/// The read back sRGB target `data` holds P1's reference composited over `background`
/// within E15's color criteria, opaque everywhere.
fn assert_resolved_over(data: &[u8], background: [f32; 3]) {
    let (reference, request_background) = p1_reference();
    assert_eq!(request_background, [0.0; 3], "P1's reference is over black");
    let (w, h) = (reference.width as usize, reference.height as usize);
    let expected = RenderImage {
        rgb: reference
            .rgb
            .iter()
            .enumerate()
            .map(|(i, c)| c + (1.0 - reference.alpha[i / 3]) * background[i % 3])
            .collect(),
        ..reference.clone()
    };
    let row = data.len() / h;
    let mut rgb = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                rgb.push(f32::from(data[y * row + 4 * x + c]) / 255.0);
            }
            assert_eq!(data[y * row + 4 * x + 3], 255, "opaque at ({x}, {y})");
        }
    }
    let ours = RenderImage {
        rgb,
        ..reference.clone()
    };
    let m = metrics(&expected, &ours);
    eprintln!(
        "P1 through the Bevy adapter (LinearOver): PSNR {:.2} dB, s {:.4}",
        m.psnr, m.share
    );
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE,
        "E15's tier: {m:?}"
    );
}

/// `FileEncoded` on an sRGB target is not drawn (D-6, Q-100): the hardware would store
/// encode(file value), the linear-light misreading R-11 rules out. Every pixel keeps the
/// camera's transparent clear, where a drawn P1 covers pixels.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_does_not_draw_file_encoded_onto_an_srgb_target() {
    let _layer = layer_messages(); // R-16, Q-105
    let (data, stats, _) = render_p1(OutputSpace::FileEncoded, TextureFormat::Rgba8UnormSrgb, 0);
    // The view was refused, so it drew nothing and counted nothing.
    assert!(
        stats.0.is_none(),
        "no statistics from a refused view: {stats:?}"
    );
    let (_, request) = probe("P1");
    let (w, h) = (
        request.items[0].camera.width as usize,
        request.items[0].camera.height as usize,
    );
    let row = data.len() / h;
    for y in 0..h {
        let pixels = &data[y * row..y * row + 4 * w];
        assert!(
            pixels.iter().all(|&b| b == 0),
            "row {y} holds only the clear value (0, 0, 0, 0)"
        );
    }
}

/// Frames rendered with nothing changed after the settle, so the statistics read are a
/// skipped frame's on any GPU fewer than this many frames behind.
const UNCHANGED_FRAMES: usize = 8;

/// A view whose camera and instances are unchanged skips stages 1–4 (Q-100) and only
/// draws: its statistics time the draw alone. The cull and sort wrote no timestamp that
/// frame, so they are 0 (Q-81), and the total runs from the draw's first timestamp to its
/// last (Q-115), on a device with timestamp queries.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_reports_only_the_draw_on_a_frame_that_skips_stages_1_to_4() {
    let _layer = layer_messages(); // R-16, Q-105
    let (_, stats, timestamps) = render_p1(
        OutputSpace::FileEncoded,
        TextureFormat::Rgba16Float,
        UNCHANGED_FRAMES,
    );
    assert_stats(stats);
    let t = stats.0.expect("statistics").gpu_times_ms;
    eprintln!("a skipped frame's GPU times: {t:?}");
    if !timestamps {
        eprintln!("no TIMESTAMP_QUERY: the GPU times are not checked");
        return;
    }
    assert_eq!((t.cull, t.sort), (0.0, 0.0), "no stage 1–4 times: {t:?}");
    assert!(t.draw > 0.0, "the draw is timed: {t:?}");
    assert_eq!(t.total, t.draw, "the total is the draw's alone: {t:?}");
}

/// §2.1: a `SplatModel` comes "with Transform (uniform scale) and Visibility", so an
/// entity spawned with only the component has both (#202).
#[test]
fn splat_model_requires_transform_and_visibility() {
    let mut world = World::new();
    let entity = world.spawn(SplatModel(bevy::asset::Handle::default())).id();
    let entity = world.entity(entity);
    assert!(
        entity.contains::<Transform>(),
        "SplatModel requires Transform"
    );
    assert!(
        entity.contains::<Visibility>(),
        "SplatModel requires Visibility"
    );
}

/// Q-99: the bevy components derive `Clone, Copy, Debug, PartialEq` where their fields
/// allow (#202). A handle and model bytes are not `Copy`.
#[test]
fn splat_components_derive_what_q99_lists_where_their_fields_allow() {
    fn copy<T: Clone + Copy + std::fmt::Debug + PartialEq>() {}
    fn no_copy<T: Clone + std::fmt::Debug + PartialEq>() {}
    copy::<SplatTime>();
    copy::<SplatMode>();
    copy::<SplatTransform>();
    copy::<SplatOverlays>();
    copy::<SplatCamera>();
    no_copy::<SplatModel>();
    no_copy::<SplatLoad>();
}

/// A `SplatCamera` in an App without a renderer, inserted in two steps when
/// `camera_later`: the component first, `Camera3d` on a later frame.
fn splat_camera_after(camera_later: bool) -> (Camera, Msaa, Tonemapping) {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        SplatPlugin {
            tier: QualityTier::conformance(),
        },
    ));
    let splat = SplatCamera {
        sort: SortMode::PerView,
        output: OutputSpace::LinearOver { background: None },
    };
    let entity = if camera_later {
        let entity = app.world_mut().spawn(splat).id();
        app.update();
        app.world_mut()
            .entity_mut(entity)
            .insert(Camera3d::default());
        entity
    } else {
        app.world_mut().spawn((Camera3d::default(), splat)).id()
    };
    app.update();
    let e = app.world().entity(entity);
    (
        e.get::<Camera>().expect("a camera").clone(),
        *e.get::<Msaa>().expect("an Msaa"),
        *e.get::<Tonemapping>().expect("a Tonemapping"),
    )
}

/// D-6, Q-100: a splat camera ends with no MSAA, no tonemapping and a transparent
/// clear whatever the order its components were inserted in (#203).
#[test]
fn splat_camera_follows_d6_whatever_the_insertion_order() {
    for camera_later in [false, true] {
        let (camera, msaa, tonemapping) = splat_camera_after(camera_later);
        assert!(
            matches!(camera.clear_color, ClearColorConfig::Custom(c) if c == Color::NONE),
            "transparent clear (camera later: {camera_later}): {:?}",
            camera.clear_color
        );
        assert_eq!(msaa, Msaa::Off, "camera later: {camera_later}");
        assert_eq!(
            tonemapping,
            Tonemapping::None,
            "camera later: {camera_later}"
        );
    }
}

/// §2.4 rule 4, §1.4's skip (#201): changing only a `LinearOver` background keeps the
/// view's frame buffers and does not run stages 1–4. Every statistics block after the
/// change times no cull and no sort, and the read back shows the new background.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_changing_only_the_background_skips_stages_1_to_4() {
    let _layer = layer_messages(); // R-16, Q-105
    let over = |background| OutputSpace::LinearOver {
        background: Some(background),
    };
    let mut scene = P1Scene::settled(over([0.2, 0.4, 0.6]), TextureFormat::Rgba8UnormSrgb);
    scene.readback(UNCHANGED_FRAMES);
    scene.take_stats();
    let background = [0.7, 0.1, 0.3];
    let camera = scene.camera;
    scene
        .app
        .world_mut()
        .get_mut::<SplatCamera>(camera)
        .expect("the splat camera")
        .output = over(background);
    let data = scene.readback(UNCHANGED_FRAMES);
    assert_resolved_over(&data, background);
    let stats = scene.take_stats();
    assert!(!stats.is_empty(), "statistics after the change");
    if !scene.timestamps() {
        eprintln!("no TIMESTAMP_QUERY: the GPU times are not checked");
        return;
    }
    for s in &stats {
        let t = s.gpu_times_ms;
        assert_eq!(
            (t.cull, t.sort),
            (0.0, 0.0),
            "stages 1–4 skipped: {stats:?}"
        );
    }
}

/// Settles P1 (`FileEncoded` on a 16-bit float target) over unchanged frames, applies
/// `change`, and checks that stages 1–4 ran again (§1.4's skip condition, §2.4 rule 2,
/// Q-100's key, #187): a statistics block after the change times the cull and the sort
/// (> 0, on a device with timestamp queries), or the read back differs from the
/// settled frame's.
fn assert_stages_1_to_4_rerun_after(what: &str, change: impl FnOnce(&mut P1Scene)) {
    let mut scene = P1Scene::settled(OutputSpace::FileEncoded, TextureFormat::Rgba16Float);
    let settled = scene.readback(UNCHANGED_FRAMES);
    scene.take_stats();
    change(&mut scene);
    let after = scene.readback(UNCHANGED_FRAMES);
    let stats = scene.take_stats();
    let timed = scene.timestamps()
        && stats
            .iter()
            .any(|s| s.gpu_times_ms.cull > 0.0 && s.gpu_times_ms.sort > 0.0);
    let redrawn = after != settled;
    eprintln!("{what}: stages 1–4 timed {timed}, read back changed {redrawn}");
    assert!(
        timed || redrawn,
        "{what}: stages 1–4 ran after the change: {stats:?}"
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_reruns_stages_1_to_4_after_a_change_of_tau() {
    let _layer = layer_messages(); // R-16, Q-105
    assert_stages_1_to_4_rerun_after("τ_m", |scene| {
        let model = scene.model;
        // Within P1's span [0, 1.9], away from the settled 0.95.
        scene
            .app
            .world_mut()
            .get_mut::<SplatTime>(model)
            .expect("the instance's time")
            .tau = 1.5;
    });
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_reruns_stages_1_to_4_after_the_camera_moves() {
    let _layer = layer_messages(); // R-16, Q-105
    assert_stages_1_to_4_rerun_after("the camera's transform", |scene| {
        let camera = scene.camera;
        scene
            .app
            .world_mut()
            .get_mut::<Transform>(camera)
            .expect("the camera's transform")
            .translation += Vec3::new(0.2, 0.0, 0.1);
    });
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_reruns_stages_1_to_4_after_the_instance_moves() {
    let _layer = layer_messages(); // R-16, Q-105
    assert_stages_1_to_4_rerun_after("the instance's transform", |scene| {
        let model = scene.model;
        scene
            .app
            .world_mut()
            .get_mut::<Transform>(model)
            .expect("the instance's transform")
            .translation += Vec3::new(0.2, 0.0, 0.1);
    });
}
