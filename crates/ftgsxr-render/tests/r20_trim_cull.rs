// SPDX-License-Identifier: Apache-2.0

//! R-20, the trim cull (§3.2; E17, §1.4 stage 2, D-56): a probe with rows inside and
//! outside, trimmed by a box and by a sphere, against the player's render of the same
//! model with the outside rows removed; a moving row drawn at a τ where its center is
//! inside and not at a τ where it is outside; `FrameStats.trimmed` equals the outside
//! count; a trim that is off renders like no trim; centers mode trimmed alike. E15's tier
//! for the renders; exact for the counts and for off.

use crate::display_support;

use display_support::{Frame, Row, Run, draw, model, probe_camera, render, upload};
use ftgsxr_core::{Camera, Model, RenderImage, Trim, TrimMode, Vec3};
use ftgsxr_render::{
    ColorTransform, Colormap, FrameBuffers, FrameStats, ModelDraw, QualityTier, RenderMode,
    SortMode, SortPose, SplatDevice, SplatPasses, ViewParams,
};
use ftgsxr_testkit::{gpu_context, layer_messages, metrics};

/// E15's tier (D-46).
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;

const TAU: f64 = 0.95;

/// Six still rows at distinct depths (y) across the probe camera's image.
fn rows() -> Vec<Row> {
    let at = [
        [-1.2, 0.0, 0.5],
        [-0.4, 0.2, -0.5],
        [0.4, 0.4, 0.3],
        [1.2, 0.6, -0.3],
        [0.0, 1.5, 0.8],
        [-0.8, -0.5, -0.9],
    ];
    let colors = [
        [0.9, 0.2, 0.1],
        [0.1, 0.8, 0.3],
        [0.2, 0.3, 0.9],
        [0.7, 0.7, 0.1],
        [0.6, 0.1, 0.7],
        [0.3, 0.9, 0.9],
    ];
    (0..6)
        .map(|k| Row {
            position: at[k],
            velocity: [0.0; 3],
            time_center: 0.95,
            duration: 100.0,
            scale: 0.15,
            opacity: 0.8,
            color: colors[k],
        })
        .collect()
}

/// The box keeps rows 0, 1, 2 and 5 (row 3 has x > 1, row 4 y > 1).
fn box_trim() -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: Vec3::new(-1.5, -1.0, -1.0),
        box_max: Vec3::new(1.0, 1.0, 1.0),
        center: Vec3::ZERO,
        radius: 1.0,
    }
}

/// The unit sphere at the origin keeps rows 1 and 2 (row 0 lies at 1.30, row 5 at 1.30).
fn sphere_trim() -> Trim {
    Trim {
        mode: TrimMode::Sphere,
        box_min: Vec3::ZERO,
        box_max: Vec3::ZERO,
        center: Vec3::ZERO,
        radius: 1.0,
    }
}

fn trimmed<'a>(d: ModelDraw<'a>, trim: Trim) -> ModelDraw<'a> {
    ModelDraw { trim, ..d }
}

/// The layer's premultiplied color and alpha as a render E15 compares.
fn image(frame: &Frame) -> RenderImage {
    RenderImage {
        width: 64,
        height: 48,
        rgb: frame
            .layer
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        alpha: frame.layer.iter().map(|p| p[3]).collect(),
    }
}

fn assert_tier(what: &str, a: &Frame, b: &Frame) {
    let m = metrics(&image(a), &image(b));
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
        "{what}: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
        m.psnr,
        100.0 * m.share,
        100.0 * m.alpha
    );
}

fn layer_only() -> Run {
    Run {
        resolve: None,
        depth: false,
    }
}

fn sort_pose(cam: &Camera) -> SortPose {
    let c2w = cam.world_to_camera.inverse();
    SortPose {
        origin: c2w.w_axis.truncate().as_vec3(),
        forward: c2w.z_axis.truncate().as_vec3().normalize(),
    }
}

/// Three frames of `draws` with the statistics; the third polls the first's block.
fn stats(dev: &SplatDevice, draws: &[ModelDraw], cam: &Camera) -> FrameStats {
    let tier = QualityTier {
        sort_mode: SortMode::PerView,
        ..QualityTier::conformance()
    };
    let passes = SplatPasses::new(dev, wgpu::TextureFormat::Rgba16Float, None, &tier)
        .expect("the passes compile");
    let capacity: u32 = 2 * draws.iter().map(|d| d.model.n()).sum::<u32>();
    let frame = FrameBuffers::new(dev, &tier, capacity);
    let views = [ViewParams {
        camera: *cam,
        target_size: ftgsxr_core::UVec2::new(cam.width, cam.height),
    }];
    let mut out = None;
    for _ in 0..3 {
        let mut enc = dev
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        passes.encode_prepare(&mut enc, &frame, draws, &sort_pose(cam), &views);
        passes.encode_stats(&mut enc, &frame);
        dev.queue().submit([enc.finish()]);
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the frame completes");
        out = FrameStats::poll(&frame, dev);
    }
    out.expect("frame 0's statistics at frame 2")
}

fn keep(rows: &[Row], kept: &[usize]) -> Model {
    let kept: Vec<Row> = kept.iter().map(|&k| rows[k]).collect();
    model(&kept)
}

/// The box and the sphere: the trimmed render matches the render of the kept rows, and
/// the outside rows are counted, in splats mode and in centers mode.
#[test]
#[ignore = "gpu"]
fn gpu_r20_box_and_sphere_trims_match_the_rows_removed() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let cam = probe_camera();
    let rows = rows();
    let all = upload(&dev, &model(&rows));
    let centers = RenderMode::Centers {
        colormap: Colormap::Dc,
        pixel_radius: 3.0,
    };
    for (name, trim, kept) in [
        ("box", box_trim(), vec![0, 1, 2, 5]),
        ("sphere", sphere_trim(), vec![1, 2]),
    ] {
        let only = upload(&dev, &keep(&rows, &kept));
        let outside = (rows.len() - kept.len()) as u32;
        for mode in [RenderMode::Splats, centers] {
            let d = trimmed(draw(&all, TAU, mode, ColorTransform::IDENTITY), trim);
            let want = render(
                &dev,
                &[draw(&only, TAU, mode, ColorTransform::IDENTITY)],
                &cam,
                &layer_only(),
            );
            let got = render(&dev, &[d], &cam, &layer_only());
            assert_tier(&format!("{name} {mode:?}"), &got, &want);
            let s = stats(&dev, &[d], &cam);
            assert_eq!(
                s.trimmed, outside,
                "{name} {mode:?}: the outside rows counted"
            );
            let visible = s.visible_splats + s.visible_centers;
            assert_eq!(
                visible,
                kept.len() as u32,
                "{name} {mode:?}: the kept rows drawn"
            );
        }
        // Both modes count a row once per instance (as the non-finite cull, Q-118).
        let both = RenderMode::Both {
            colormap: Colormap::Dc,
            pixel_radius: 3.0,
        };
        let s = stats(
            &dev,
            &[trimmed(
                draw(&all, TAU, both, ColorTransform::IDENTITY),
                trim,
            )],
            &cam,
        );
        assert_eq!(s.trimmed, outside, "{name} both: once per row");
    }
    assert!(dev.take_validation_messages().is_empty());
}

/// A moving row is drawn at a τ where its center lies inside the trim, and trimmed at a
/// τ where it lies outside.
#[test]
#[ignore = "gpu"]
fn gpu_r20_a_moving_row_follows_its_center() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let cam = probe_camera();
    // x = τ − 0.95: inside x ≤ 0.5 at τ = 0.95, outside at τ = 1.8 (x = 0.85).
    let row = Row {
        position: [0.0; 3],
        velocity: [1.0, 0.0, 0.0],
        time_center: 0.95,
        duration: 100.0,
        scale: 0.15,
        opacity: 0.8,
        color: [0.9, 0.5, 0.1],
    };
    let m = upload(&dev, &model(&[row]));
    let trim = Trim {
        mode: TrimMode::Box,
        box_min: Vec3::splat(-0.5),
        box_max: Vec3::splat(0.5),
        center: Vec3::ZERO,
        radius: 1.0,
    };
    let plain = |tau| draw(&m, tau, RenderMode::Splats, ColorTransform::IDENTITY);
    let inside = render(&dev, &[trimmed(plain(0.95), trim)], &cam, &layer_only());
    let untrimmed = render(&dev, &[plain(0.95)], &cam, &layer_only());
    assert_tier("inside", &inside, &untrimmed);
    assert!(inside.layer.iter().any(|p| p[3] > 0.1), "the row is drawn");
    assert_eq!(stats(&dev, &[trimmed(plain(0.95), trim)], &cam).trimmed, 0);

    let outside = render(&dev, &[trimmed(plain(1.8), trim)], &cam, &layer_only());
    assert!(
        outside.layer.iter().all(|p| *p == [0.0; 4]),
        "nothing drawn once the center leaves the trim"
    );
    let s = stats(&dev, &[trimmed(plain(1.8), trim)], &cam);
    assert_eq!((s.trimmed, s.visible_splats), (1, 0));
    // Untrimmed, the row is on the image at τ = 1.8.
    assert!(
        render(&dev, &[plain(1.8)], &cam, &layer_only())
            .layer
            .iter()
            .any(|p| p[3] > 0.1)
    );
    assert!(dev.take_validation_messages().is_empty());
}

/// A trim that is off renders exactly as no trim, whatever its shapes, and counts none.
#[test]
#[ignore = "gpu"]
fn gpu_r20_a_trim_that_is_off_renders_like_no_trim() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let cam = probe_camera();
    let all = upload(&dev, &model(&rows()));
    let plain = draw(&all, TAU, RenderMode::Splats, ColorTransform::IDENTITY);
    let off = Trim {
        mode: TrimMode::Off,
        ..sphere_trim()
    };
    let a = render(&dev, &[plain], &cam, &layer_only());
    let b = render(&dev, &[trimmed(plain, off)], &cam, &layer_only());
    assert_eq!(a.layer, b.layer, "bit for bit");
    let (sa, sb) = (
        stats(&dev, &[plain], &cam),
        stats(&dev, &[trimmed(plain, off)], &cam),
    );
    assert_eq!((sb.trimmed, sb.visible_splats), (0, sa.visible_splats));
    assert!(dev.take_validation_messages().is_empty());
}
