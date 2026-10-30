// SPDX-License-Identifier: Apache-2.0

//! Shared helpers of the cull tests (R-5, R-6): random cameras rounded to float32, the
//! upload, one frame through `encode_prepare`, and the CPU oracle built on
//! `ftgsxr_core::project` (E1, E2, E5, E6 in float64).

#![allow(dead_code, reason = "each test file uses a part of the helpers")]

use ftgsxr_core::{
    Camera, DMat3, DMat4, Mat3, Model, Pose, Projected, Quat, TemporalIndex, UVec2, Vec3,
    covariance, effective_opacity, moved_position, pack_for_gpu, posed_covariance, project,
    quat_from_wxyz, temporal_opacity,
};
use ftgsxr_render::{
    ColorTransform, FrameBuffers, FrameContents, GpuModel, ModelDraw, Overlays, QualityTier,
    RenderMode, SortPose, SplatDevice, SplatPasses, ViewParams,
};

/// SplitMix64, the tests' own seeded generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }

    /// A uniformly random unit vector.
    pub fn unit(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.range(-1.0, 1.0) as f32,
                self.range(-1.0, 1.0) as f32,
                self.range(-1.0, 1.0) as f32,
            );
            let l = v.length();
            if (0.1..=1.0).contains(&l) {
                return v / l;
            }
        }
    }
}

/// A pinhole camera in openftgs's conventions (E6) at `eye` looking at `target`, with the
/// stage's y up, its numbers rounded to float32 first so the GPU (float32) and the oracle
/// (float64) see the same camera.
pub fn look_at(eye: Vec3, target: Vec3, f: f32, size: UVec2, principal: [f32; 2]) -> Camera {
    let z = (target - eye).normalize();
    let x = z.cross(Vec3::Y).normalize();
    let y = z.cross(x);
    let r_w = Mat3::from_cols(x, y, z).transpose();
    let t_w = -(r_w * eye);
    let mut w2c = DMat4::from_mat3(r_w.as_dmat3());
    w2c.w_axis = t_w.as_dvec3().extend(1.0);
    let k = DMat3::from_cols(
        ftgsxr_core::DVec3::new(f64::from(f), 0.0, 0.0),
        ftgsxr_core::DVec3::new(0.0, f64::from(f), 0.0),
        ftgsxr_core::DVec3::new(f64::from(principal[0]), f64::from(principal[1]), 1.0),
    );
    Camera {
        width: size.x,
        height: size.y,
        k,
        world_to_camera: w2c,
    }
}

/// The camera's center and forward axis, −R_wᵀ t_w and R_wᵀ ẑ.
pub fn sort_pose(cam: &Camera) -> SortPose {
    let r = DMat3::from_mat4(cam.world_to_camera);
    let t = cam.world_to_camera.w_axis.truncate();
    SortPose {
        origin: (-(r.transpose() * t)).as_vec3(),
        forward: (r.transpose() * ftgsxr_core::DVec3::Z).as_vec3(),
    }
}

pub fn view(cam: Camera) -> ViewParams {
    ViewParams {
        camera: cam,
        target_size: UVec2::new(cam.width, cam.height),
    }
}

/// `model` packed in its temporal index's order and uploaded under `tier`'s layout.
pub fn upload(dev: &SplatDevice, model: &Model, tier: &QualityTier) -> GpuModel {
    let (index, order) = TemporalIndex::build(model);
    let arrays = pack_for_gpu(model, &order, &tier.layout);
    GpuModel::upload(dev, &arrays, index, &model.conventions).expect("fits the budget")
}

pub fn draw<'a>(model: &'a GpuModel, pose: Pose, tau: f64, mode: RenderMode) -> ModelDraw<'a> {
    ModelDraw {
        model,
        pose,
        tau,
        mode,
        transform: ColorTransform { gain: Vec3::ONE },
        sh_degree: 3,
        overlays: Overlays::default(),
    }
}

/// One frame of stages 1–3 for `draws`, submitted and read back (Q-69).
pub fn prepare(
    dev: &SplatDevice,
    passes: &SplatPasses,
    frame: &FrameBuffers,
    draws: &[ModelDraw],
    pose: &SortPose,
    views: &[ViewParams],
) -> FrameContents {
    let mut enc = dev
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    passes.encode_prepare(&mut enc, frame, draws, pose, views);
    dev.queue().submit([enc.finish()]);
    frame.read_back(dev)
}

/// The conformance tier's passes for a 16-bit float layer (§1.4 stage 5).
pub fn passes(dev: &SplatDevice, tier: &QualityTier) -> SplatPasses {
    SplatPasses::new(dev, wgpu::TextureFormat::Rgba16Float, None, tier).expect("the passes compile")
}

/// A row of `model` at τ in stage space: σ_eff (E2, float64 then rounded, U-7's
/// `effective_opacity`), μ^S (E1, E5) and Σ^S (E5).
pub fn row_at(model: &Model, i: usize, pose: &Pose, tau: f64) -> (f32, Vec3, Mat3) {
    let single = model.is_single_timestamp();
    let sigma_t = temporal_opacity(tau, model.time_center[i], model.duration[i], single);
    let sigma_eff = effective_opacity(model.opacity[i], sigma_t);
    let mu = if single {
        Vec3::from(model.position[i])
    } else {
        moved_position(
            Vec3::from(model.position[i]),
            Vec3::from(model.velocity[i]),
            model.time_center[i],
            tau,
        )
    };
    let r = pose.rotation.as_dquat();
    let mu_s = (r * mu.as_dvec3() * f64::from(pose.scale) + pose.translation.as_dvec3()).as_vec3();
    let cov = posed_covariance(
        covariance(
            quat_from_wxyz(model.rotation[i]),
            Vec3::from(model.scale[i]),
        ),
        pose,
    );
    (sigma_eff, mu_s, cov)
}

/// The CPU oracle of one view: every row `project` keeps (E6), with its file row.
pub fn oracle(model: &Model, pose: &Pose, tau: f64, cam: &Camera) -> Vec<(usize, Projected)> {
    (0..model.len())
        .filter_map(|i| {
            let (sigma_eff, mu_s, cov) = row_at(model, i, pose, tau);
            if sigma_eff < model.conventions.alpha_skip {
                return None;
            }
            project(mu_s, cov, sigma_eff, cam, &model.conventions).map(|p| (i, p))
        })
        .collect()
}

/// A pose with a rotation, a scale and a translation (E5).
pub fn test_pose() -> Pose {
    Pose {
        translation: Vec3::new(0.3, 1.2, -0.4),
        rotation: Quat::from_axis_angle(Vec3::new(1.0, 2.0, 0.5).normalize(), 0.7),
        scale: 1.3,
    }
}
