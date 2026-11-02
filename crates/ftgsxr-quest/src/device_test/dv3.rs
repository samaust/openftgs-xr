// SPDX-License-Identifier: Apache-2.0

//! Dv-3 on the device (§3.4, Q-122): R-4's criteria (§3.2; §1.4 stages 3 and 4, E8, D-33)
//! on the Quest's GPU, through the headless renderer's own device (`SplatDevice::headless`,
//! Q-74). The host driver pushed two exchange files written with `write_exchange` (Q-65):
//! `dv/r4-random.safetensors`, 10⁶ rows at log-uniform random depths, and
//! `dv/r4-repeated.safetensors`, 10⁶ rows over 1000 distinct depths, every row visible
//! from [`camera`] at τ = [`TAU`]. Three checks, as R-4's three tests: the random model in
//! the `conformance` tier, the repeated one in it (stability), and the random one with
//! `Depth16` (its top 16 bits, Q-2). Each reads the frame back (Q-69), recomputes each
//! slot's key from its record's depth with `sort_key` (`sort_key16` for `Depth16`) and
//! requires `sorted` to equal the slots stably sorted by that key, every row visible, no
//! overflow and the sizing pass's dispatch arguments ⌈count / 1024⌉ (Q-85).

use std::path::Path;

use bevy::render::render_resource::{CommandEncoderDescriptor, TextureFormat};
// Q-T226.1
use ftgsxr_app::{
    Camera, ColorTransform, DMat3, DMat4, DVec3, FrameBuffers, FrameContents, GpuModel, ModelDraw,
    Overlays, Pose, QualityTier, RenderMode, SortKey, SortPose, SplatDevice, SplatPasses,
    TemporalIndex, Vec3, ViewParams, open_model, pack_for_gpu, sort_key, sort_key16,
};

/// R-4's size; the driver writes this many rows.
const N: u32 = 1_000_000;
/// The keys one workgroup of the sort's count pass takes (Q-85).
const SORT_BLOCK: u32 = 1024;
/// The frame's time: every row's time center.
const TAU: f64 = 0.95;

/// A camera at the origin looking along +z, 800 × 800 px, f = 400 px, principal point
/// (400, 400): the driver places every row inside its image (|x|, |y| ≤ 0.4 z).
fn camera() -> Camera {
    Camera {
        width: 800,
        height: 800,
        k: DMat3::from_cols(
            DVec3::new(400.0, 0.0, 0.0),
            DVec3::new(0.0, 400.0, 0.0),
            DVec3::new(400.0, 400.0, 1.0),
        ),
        world_to_camera: DMat4::IDENTITY,
    }
}

/// One frame of the model at `path` under `tier`, read back.
fn frame(dev: &SplatDevice, path: &Path, tier: &QualityTier) -> Result<FrameContents, String> {
    let model = open_model(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (index, order) = TemporalIndex::build(&model);
    let arrays = pack_for_gpu(&model, &order, &tier.layout);
    let gpu_model = GpuModel::upload(dev, &arrays, index, &model.conventions)
        .map_err(|e| format!("upload: {}: {}", e.code(), e.user_text()))?;
    let passes = SplatPasses::new(dev, TextureFormat::Rgba16Float, None, tier)
        .map_err(|e| format!("passes: {}: {}", e.code(), e.user_text()))?;
    let frame = FrameBuffers::new(dev, tier, N);
    let cam = camera();
    let draws = [ModelDraw {
        model: &gpu_model,
        pose: Pose::IDENTITY,
        tau: TAU,
        mode: RenderMode::Splats,
        transform: ColorTransform { gain: Vec3::ONE },
        sh_degree: 3,
        overlays: Overlays::default(),
    }];
    let pose = SortPose {
        origin: Vec3::ZERO,
        forward: Vec3::Z,
    };
    let views = [ViewParams {
        camera: cam,
        target_size: ftgsxr_app::UVec2::new(cam.width, cam.height),
    }];
    let mut enc = dev
        .device()
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    passes.encode_prepare(&mut enc, &frame, &draws, &pose, &views);
    dev.queue().submit([enc.finish()]);
    let got = frame.read_back(dev);
    let messages = dev.take_validation_messages();
    if let Some(m) = messages.first() {
        return Err(format!("GPU validation: {m}"));
    }
    Ok(got)
}

/// R-4's criteria on `got` with each slot's key from its record's depth, `digits` of it.
fn check(got: &FrameContents, digits: impl Fn(u32) -> u32) -> Result<(), String> {
    if got.visible_splats != N {
        return Err(format!(
            "{} of {N} rows visible ({} active rows)",
            got.visible_splats, got.active_rows
        ));
    }
    if got.overflow {
        return Err("overflow".into());
    }
    let dispatch = [got.visible_splats.div_ceil(SORT_BLOCK), 1, 1];
    if got.sort_dispatch != dispatch {
        return Err(format!(
            "sort dispatch {:?}, expected {dispatch:?}",
            got.sort_dispatch
        ));
    }
    let keys = got
        .records
        .iter()
        .enumerate()
        .map(|(slot, r)| {
            sort_key(r.depth)
                .map(&digits)
                .ok_or_else(|| format!("slot {slot}: depth {} has no key", r.depth))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    // The slots stably sorted by key (`sort_by_key` is stable).
    let mut expected: Vec<u32> = (0..keys.len() as u32).collect();
    expected.sort_by_key(|&slot| keys[slot as usize]);
    if got.sorted.len() != expected.len() {
        return Err(format!("{} sorted slots", got.sorted.len()));
    }
    if let Some(i) = got.sorted.iter().zip(&expected).position(|(a, b)| a != b) {
        return Err(format!(
            "position {i}: slot {} (key {:#x}), the stable CPU sort has slot {} (key {:#x})",
            got.sorted[i],
            keys.get(got.sorted[i] as usize).copied().unwrap_or(0),
            expected[i],
            keys[expected[i] as usize]
        ));
    }
    Ok(())
}

/// The three checks; `Ok` with a summary, or the first disagreement.
pub(super) fn run(dv: &Path) -> Result<String, String> {
    let dev = SplatDevice::headless().map_err(|e| format!("{}: {}", e.code(), e.user_text()))?;
    let random = dv.join("r4-random.safetensors");
    let repeated = dv.join("r4-repeated.safetensors");
    let conformance = QualityTier::conformance();
    check(&frame(&dev, &random, &conformance)?, |k| k).map_err(|e| format!("random: {e}"))?;
    check(&frame(&dev, &repeated, &conformance)?, |k| k)
        .map_err(|e| format!("repeated keys: {e}"))?;
    let depth16 = QualityTier {
        sort_key: SortKey::Depth16,
        ..QualityTier::conformance()
    };
    check(&frame(&dev, &random, &depth16)?, |k| {
        u32::from(sort_key16(k))
    })
    .map_err(|e| format!("Depth16: {e}"))?;
    Ok(format!(
        "{N} keys sorted stably in three frames on {}",
        dev.adapter_name()
    ))
}
