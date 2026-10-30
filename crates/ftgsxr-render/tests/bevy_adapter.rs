// SPDX-License-Identifier: Apache-2.0

//! PRD #76's added test, the Bevy adapter renders like the headless renderer (source of
//! truth: R-1's criterion, §3.2): a Bevy `App` with `SplatPlugin { tier: conformance }`,
//! one `SplatCamera` with `OutputSpace::FileEncoded` rendering into an image target and
//! one `SplatModel` of P1 at the request's first item, read back, meets E15's tier
//! (PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %) against the reference render. It catches a broken
//! plugin wiring (loads, uploads, extraction, the two render nodes, the camera's
//! conversion to E6's camera); window and swapchain behaviour are M-4's I-n.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::{Mat3, Quat, Vec3};
use bevy::mesh::MeshPlugin;
use bevy::render::RenderPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_core::{Camera as FtgsCamera, DMat3, RenderImage};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad, SplatPlugin,
    SplatStats, SplatTime,
};
use ftgsxr_testkit::{fixtures_dir, metrics, probe, reference_render};

/// E15's tier.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;
/// Frames the app may take to load, upload and draw before the test gives up.
const MAX_FRAMES: usize = 600;
/// Frames rendered after the upload before the read back is taken, so the read back
/// holds a frame that drew the model.
const SETTLE_FRAMES: usize = 4;

/// Set by the render error handler: Bevy reported a validation, out-of-memory or
/// internal error, or lost the device (§1.13: any message fails a `gpu` test).
static RENDER_ERROR: AtomicBool = AtomicBool::new(false);

#[derive(Resource, Clone, Default)]
struct Seen {
    events: Arc<Mutex<Vec<SplatEvent>>>,
    readback: Arc<Mutex<Option<Vec<u8>>>>,
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

/// P1's first item drawn by a Bevy `App` with `SplatPlugin { tier: conformance }`: the
/// model loaded through `SplatLoad` at the item's timestamp, one `SplatCamera` writing
/// `output` into an image target of `format`; returns the target's bytes as read back,
/// a few frames after the upload, and the `SplatStats` then.
fn render_p1(output: OutputSpace, format: TextureFormat) -> (Vec<u8>, SplatStats) {
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
    .insert_resource(RenderErrorHandler(|error, _, _| {
        eprintln!("render error: {:?}: {}", error.ty, error.description);
        RENDER_ERROR.store(true, Ordering::Release);
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

    let mut image = Image::new_target_texture(w, h, format, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);

    app.world_mut().spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(target.clone().into()),
        bevy_projection(&item.camera),
        bevy_transform(&item.camera),
        SplatCamera {
            sort: SortMode::PerView,
            output,
        },
    ));
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

    let mut uploaded_at = None;
    for frame in 0..MAX_FRAMES {
        app.update();
        assert!(
            !RENDER_ERROR.load(Ordering::Acquire),
            "Bevy reported a render error"
        );
        let events = seen.events.lock().unwrap().clone();
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
            // A fresh read back, from a frame after the settle.
            *seen.readback.lock().unwrap() = None;
            app.update();
            app.update();
            break;
        }
    }
    assert!(
        uploaded_at.is_some(),
        "P1 uploaded within {MAX_FRAMES} frames"
    );
    assert!(
        !RENDER_ERROR.load(Ordering::Acquire),
        "Bevy reported a render error"
    );
    let data = seen
        .readback
        .lock()
        .unwrap()
        .clone()
        .expect("a read back of the image target");
    let stats = *app.world().resource::<SplatStats>();
    (data, stats)
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
    // The image target is 16-bit float RGBA, so the view's main texture is the layer's
    // format and `FileEncoded` holds the layer unrounded.
    let (data, stats) = render_p1(OutputSpace::FileEncoded, TextureFormat::Rgba16Float);
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
    let background = [0.2, 0.4, 0.6];
    let (data, stats) = render_p1(
        OutputSpace::LinearOver {
            background: Some(background),
        },
        TextureFormat::Rgba8UnormSrgb,
    );
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
    assert_stats(stats);
}
