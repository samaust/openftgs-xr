// SPDX-License-Identifier: Apache-2.0

//! Q-450's tightened splat quad (§1.4 stage 5, revision 61) at a render scale below 1
//! (§1.12, E16; #825). In `quad_pixel` the factor `camera_size/target_size` scales δ, the
//! pad and the margin m. `HeadlessRenderer` always draws at target = camera size (Q-109),
//! so Q-448's layers never see that factor away from 1. This test draws the splat layer
//! through `SplatPasses::encode_draw` itself, at a `target_size` below the camera's, on a
//! device it makes and wraps with `SplatDevice::from_wgpu`, as Q-451's test does. It
//! compares the layers by E15's rule (s = 0 and s\_α = 0), as Q-445's and Q-448's checks
//! do, on every adapter, with layers recorded at 1a95a298, the commit before the tightened
//! quad (Q-453).
//!
//! The scenes are R2\_random's turned eye pairs (Q-448's cameras, `turned_eyes`), per view
//! and in shared-center mode in both record layouts. They are drawn at `pixel_center_offset`
//! 0.5 and 0, at `screen_extent_max` 2, and at `pixel_center_offset` 0 with `alpha_skip`
//! 0.05, at render scales 0.7 and E16's floor 0.6: 144 layers. The target size is the Bevy
//! adapter's `layer_size`: each side × the scale, rounded, at least 1 (Q-453). The
//! `alpha_skip` 0.05 layers raise the ellipse's rim, where the scale dropped from δ in
//! `quad_pixel` drops texels, above E15's threshold at both scales (#827, measured on an
//! RTX 4090).

use std::path::PathBuf;

use crate::cull_support::{sort_pose, upload};
use crate::q448_exact::turned_eyes;

use ftgsxr_core::{Camera, Pose, RenderConventions, RenderImage, UVec2};
use ftgsxr_render::{
    ColorTransform, FrameBuffers, ModelDraw, Overlays, QualityTier, RenderMode, SortMode, SortPose,
    SplatDevice, SplatPasses, ViewParams,
};
use ftgsxr_testkit::{layer_messages, metrics, probe};

/// The recorded layers (`scaled_layers`), after the recording adapter's name, which is kept
/// for the record only (Q-453).
fn recorded_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/q450_render_scale.bin")
}

/// Set to record `scaled_layers` into `recorded_path()` instead of comparing, only under
/// Q-445's rule as written: in a commit of its own, with the shaders of the commit before
/// an exact change (Q-453; #828 recorded with 1a95a298's `splat.wgsl` in the current tree).
const RECORD_VAR: &str = "FTGSXR_RECORD_Q450_SCALED";

/// The layer format, the headless renderer's (Q-109): four half floats.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The bytes of one texel in `FORMAT`.
const TEXEL_BYTES: u32 = 8;

/// The render scales drawn: 0.7, and E16's floor 0.6, which the governor steps down to
/// under load (Q-453).
// Q-453
const SCALES: [(&str, f32); 2] = [("0.7", 0.7), ("0.6", 0.6)];

/// A variant's id suffix and its change to the model's conventions.
type Variant = (&'static str, fn(&mut RenderConventions));

/// R2_random's conventions at `pixel_center_offset` 0.5 (δ = 0, so only the scaled pad and
/// m act), at 0, where δ = −½ × `camera_size/target_size`, at `screen_extent_max` 2,
/// where E6's cap binds and the pixels it clips reach α = σ_eff·e⁻² (#820), and at
/// `pixel_center_offset` 0 with `alpha_skip` 0.05, where the rim texels a misplaced quad
/// drops reach α ≈ 0.05, above E15's threshold (Q-453, #827).
// Q-453
const VARIANTS: [Variant; 4] = [
    ("-offset0.5", |c| c.pixel_center_offset = 0.5),
    ("-offset0", |c| c.pixel_center_offset = 0.0),
    ("-cap2", |c| c.screen_extent_max = 2.0),
    ("-offset0-skip0.05", |c| {
        c.pixel_center_offset = 0.0;
        c.alpha_skip = 0.05;
    }),
];

/// The Bevy adapter's splat-layer size at render scale `scale` (`layer_size`, §1.12):
/// each side × the scale, rounded, at least 1 (Q-453).
// Q-453
fn scaled_size(camera: &Camera, scale: f32) -> UVec2 {
    let side = |n: u32| ((n as f32 * scale).round() as u32).max(1);
    UVec2::new(side(camera.width), side(camera.height))
}

/// A device of the test's own, wrapped with `SplatDevice::from_wgpu` (as Q-451's is), or
/// why there is none: no Vulkan adapter or device. The features are
/// `SplatDevice::headless`'s, where offered.
fn scaled_device() -> Result<SplatDevice, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
    let instance = wgpu::Instance::new(desc);
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|e| format!("no Vulkan adapter: {e}"))?;
    let info = adapter.get_info();
    let wanted = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ftgsxr q450 render-scale device"),
        required_features: adapter.features() & wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("{}: no device: {e}", info.name))?;
    Ok(SplatDevice::from_wgpu(device, queue, info))
}

/// One layer: its size and its texels' bits, row-major, four half floats in a `u64`.
struct Layer {
    size: UVec2,
    texels: Vec<u64>,
}

/// A cleared layer of `view`'s target size with `view` drawn into it, and the buffer it is
/// copied into, with its row pitch.
fn draw_layer(
    dev: &SplatDevice,
    enc: &mut wgpu::CommandEncoder,
    passes: &SplatPasses,
    frame: &FrameBuffers,
    view: &ViewParams,
) -> (wgpu::Buffer, u32) {
    let device = dev.device();
    let size = wgpu::Extent3d {
        width: view.target_size.x,
        height: view.target_size.y,
        depth_or_array_layers: 1,
    };
    let layer = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("q450 layer"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let layer_view = layer.create_view(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("q450 draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        passes.encode_draw(&mut pass, frame, view, None);
    }
    let row_bytes = (size.width * TEXEL_BYTES).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q450 read back"),
        size: u64::from(row_bytes) * u64::from(size.height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &layer,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    (staging, row_bytes)
}

/// The two eyes' layers of `draws` at the views `views` in `tier`'s sort mode, after one prepare per eye
/// at its own pose (per view) or one at the eyes' midpoint with both as cull views (shared
/// center), as the Bevy adapter draws them.
fn draw_eyes(
    dev: &SplatDevice,
    passes: &SplatPasses,
    tier: &QualityTier,
    draws: &[ModelDraw],
    views: &[ViewParams; 2],
) -> [Layer; 2] {
    let capacity = 2 * draws.iter().map(|d| d.model.n()).sum::<u32>();
    let device = dev.device();
    let mut enc = device.create_command_encoder(&Default::default());
    let frames: Vec<FrameBuffers> = match tier.sort_mode {
        SortMode::PerView => views
            .iter()
            .map(|v| {
                let frame = FrameBuffers::new(dev, tier, capacity);
                passes.encode_prepare(
                    &mut enc,
                    &frame,
                    draws,
                    &sort_pose(&v.camera),
                    std::slice::from_ref(v),
                );
                frame
            })
            .collect(),
        SortMode::SharedCenter => {
            let frame = FrameBuffers::new(dev, tier, capacity);
            let [l, r] = views.map(|v| sort_pose(&v.camera));
            let midpoint = SortPose {
                origin: (l.origin + r.origin) / 2.0,
                forward: (l.forward + r.forward).normalize(),
            };
            passes.encode_prepare(&mut enc, &frame, draws, &midpoint, views);
            vec![frame]
        }
    };
    let staged: Vec<(wgpu::Buffer, u32)> = views
        .iter()
        .enumerate()
        .map(|(i, view)| {
            draw_layer(
                dev,
                &mut enc,
                passes,
                &frames[i.min(frames.len() - 1)],
                view,
            )
        })
        .collect();
    dev.queue().submit([enc.finish()]);
    for (buffer, _) in &staged {
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("the layer maps"));
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
    let layers: Vec<Layer> = staged
        .iter()
        .zip(views)
        .map(|((buffer, row_bytes), view)| {
            let bytes = buffer.slice(..).get_mapped_range();
            let (w, h) = (view.target_size.x as usize, view.target_size.y as usize);
            let texels = (0..h)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .map(|(y, x)| {
                    let at = y * *row_bytes as usize + x * TEXEL_BYTES as usize;
                    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
                })
                .collect();
            Layer {
                size: view.target_size,
                texels,
            }
        })
        .collect();
    layers.try_into().unwrap_or_else(|_| panic!("two eyes"))
}

/// The conformance tier sorting `sort_mode`, with the shared-center records' Σ^S in half
/// precision or not (§1.12 `half_records`; per view ignores the flag).
fn tier(sort_mode: SortMode, half_records: bool) -> QualityTier {
    QualityTier {
        sort_mode,
        half_records,
        ..QualityTier::conformance()
    }
}

/// The layers of R2_random's turned pairs, by id: per view and shared-center in both
/// layouts, each variant, each render scale, each item, each eye.
fn scaled_layers(dev: &SplatDevice) -> Vec<(String, Layer)> {
    let mut out = Vec::new();
    let modes = [
        ("per-view", tier(SortMode::PerView, false)),
        ("shared-full", tier(SortMode::SharedCenter, false)),
        ("shared-half", tier(SortMode::SharedCenter, true)),
    ];
    for (mode, mode_tier) in &modes {
        let passes = SplatPasses::new(dev, FORMAT, None, mode_tier).expect("the passes compile");
        for (variant, conventions) in VARIANTS {
            let (mut model, request) = probe("R2_random");
            conventions(&mut model.conventions);
            let gpu_model = upload(dev, &model, mode_tier);
            for item in &request.items {
                let draw = ModelDraw {
                    model: &gpu_model,
                    pose: Pose::IDENTITY,
                    tau: item.timestamp,
                    mode: RenderMode::Splats,
                    transform: ColorTransform {
                        gain: ftgsxr_core::Vec3::ONE,
                    },
                    sh_degree: request.sh_degree.unwrap_or(model.sh_degree),
                    overlays: Overlays::default(),
                    trim: ftgsxr_core::Trim::default(),
                };
                let eyes = turned_eyes(&item.camera);
                for (scale_name, scale) in SCALES {
                    let views = eyes.map(|camera| ViewParams {
                        camera,
                        target_size: scaled_size(&camera, scale),
                    });
                    let layers = draw_eyes(dev, &passes, mode_tier, &[draw], &views);
                    for (eye, layer) in ["L", "R"].into_iter().zip(layers) {
                        out.push((
                            format!(
                                "{mode}/R2_random-turned{variant}/scale{scale_name}/{}{eye}",
                                item.id
                            ),
                            layer,
                        ));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The f32 value of the half float `bits` (IEEE 754 binary16).
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024.0 + mantissa) * 2f32.powi(exponent - 25),
    }
}

/// `layer` as a `RenderImage` for E15's metrics: each texel's half floats (R, G, B, A, low
/// bits first) as f32.
fn image(layer: &Layer) -> RenderImage {
    let channel = |t: u64, c: u32| half_to_f32((t >> (16 * c)) as u16);
    RenderImage {
        width: layer.size.x,
        height: layer.size.y,
        rgb: layer
            .texels
            .iter()
            .flat_map(|&t| [channel(t, 0), channel(t, 1), channel(t, 2)])
            .collect(),
        alpha: layer.texels.iter().map(|&t| channel(t, 3)).collect(),
    }
}

/// The encoding of `recorded_path()`: the adapter's name, then each layer's id, its size,
/// its first texel as the fill, then each texel that differs from the fill, by index.
fn encode(adapter: &str, layers: &[(String, Layer)]) -> Vec<u8> {
    let mut out = Vec::new();
    let put = |w: u32, out: &mut Vec<u8>| out.extend_from_slice(&w.to_le_bytes());
    put(adapter.len() as u32, &mut out);
    out.extend_from_slice(adapter.as_bytes());
    put(layers.len() as u32, &mut out);
    for (id, layer) in layers {
        put(id.len() as u32, &mut out);
        out.extend_from_slice(id.as_bytes());
        put(layer.size.x, &mut out);
        put(layer.size.y, &mut out);
        let fill = layer.texels[0];
        out.extend_from_slice(&fill.to_le_bytes());
        let set: Vec<(usize, u64)> = layer
            .texels
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, t)| t != fill)
            .collect();
        put(set.len() as u32, &mut out);
        for (i, t) in set {
            put(i as u32, &mut out);
            out.extend_from_slice(&t.to_le_bytes());
        }
    }
    out
}

/// The adapter's name and the layers of `encode`.
fn decode(bytes: &[u8]) -> (String, Vec<(String, Layer)>) {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| {
        let slice = &bytes[*at..*at + n];
        *at += n;
        slice
    };
    let word = |at: &mut usize| u32::from_le_bytes(take(at, 4).try_into().expect("four bytes"));
    let texel = |at: &mut usize| u64::from_le_bytes(take(at, 8).try_into().expect("eight bytes"));
    let string = |at: &mut usize| {
        let len = word(at) as usize;
        String::from_utf8(take(at, len).to_vec()).expect("UTF-8")
    };
    let adapter = string(&mut at);
    let n = word(&mut at);
    let mut out = Vec::new();
    for _ in 0..n {
        let id = string(&mut at);
        let size = UVec2::new(word(&mut at), word(&mut at));
        let mut texels = vec![texel(&mut at); (size.x * size.y) as usize];
        for _ in 0..word(&mut at) {
            let i = word(&mut at) as usize;
            texels[i] = texel(&mut at);
        }
        out.push((id, Layer { size, texels }));
    }
    assert_eq!(at, bytes.len(), "the recorded file ends after its layers");
    (adapter, out)
}

/// Q-450, Q-453 (#825): at render scales 0.7 and 0.6 the tightened quad changes no texel of
/// R2_random's turned pairs, per view and shared-center in both record layouts, at
/// `pixel_center_offset` 0.5 and 0, at `screen_extent_max` 2 and at `pixel_center_offset` 0
/// with `alpha_skip` 0.05, against the layers recorded at the commit before it, by E15's
/// rule (s = 0 and s_α = 0, as Q-445 and Q-448) on every adapter. A failure lists the
/// texels that differ bitwise beside E15's s and s_α. With `FTGSXR_RECORD_Q450_SCALED` set
/// it records them instead.
#[test]
#[ignore = "gpu"]
fn gpu_q450_scaled_quad_changes_no_texel_of_the_recorded_layers() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = match scaled_device() {
        Ok(dev) => dev,
        Err(why) => {
            eprintln!("skipped: {why} (Q-453)");
            return;
        }
    };
    let adapter = dev.adapter_name().to_owned();
    let recording = std::env::var_os(RECORD_VAR).is_some_and(|v| !v.is_empty());
    let recorded = (!recording).then(|| {
        let bytes = std::fs::read(recorded_path()).expect("the recorded layers (Q-453)");
        decode(&bytes)
    });
    if let Some((then, _)) = &recorded {
        eprintln!("the layers were recorded on {then}; this adapter is {adapter} (Q-453)");
    }
    let layers = scaled_layers(&dev);
    let messages = dev.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:?}");
    let Some((_, recorded)) = recorded else {
        std::fs::write(recorded_path(), encode(&adapter, &layers))
            .expect("the recorded layers written");
        return;
    };
    let ids = |l: &[(String, Layer)]| l.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&layers), ids(&recorded), "the same scenes");
    let drawn = layers
        .iter()
        .filter(|(_, l)| l.texels.iter().any(|&t| t != 0))
        .count();
    assert!(drawn * 2 > layers.len(), "most layers draw something");
    let failures: Vec<String> = layers
        .iter()
        .zip(&recorded)
        .filter_map(|((id, now), (_, then))| {
            assert_eq!(now.size, then.size, "{id}: the same size");
            let differ = now
                .texels
                .iter()
                .zip(&then.texels)
                .filter(|(a, b)| a != b)
                .count();
            let m = metrics(&image(then), &image(now));
            // Q-453: E15's rule on every adapter, as Q-445 and Q-448.
            (m.share != 0.0 || m.alpha != 0.0).then(|| {
                format!(
                    "{id}: {differ} texels differ bitwise; by E15's rule s = {}, s_α = {}",
                    m.share, m.alpha
                )
            })
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} layers differ from the recorded ones by E15's rule (Q-450, Q-453):\n{}",
        failures.len(),
        layers.len(),
        failures.join("\n")
    );
}

/// `half_to_f32` decodes binary16 normals, subnormals, signs and the largest finite value.
#[test]
fn half_to_f32_decodes_binary16() {
    assert_eq!(half_to_f32(0x3c00), 1.0);
    assert_eq!(half_to_f32(0xc000), -2.0);
    assert_eq!(half_to_f32(0x3555), 0.333_251_95);
    assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(half_to_f32(0x7bff), 65504.0);
    assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    assert!(half_to_f32(0x7e00).is_nan());
    assert_eq!(half_to_f32(0x0000), 0.0);
}
