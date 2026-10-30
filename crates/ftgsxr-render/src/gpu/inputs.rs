// SPDX-License-Identifier: Apache-2.0

//! The frame inputs of `SplatPasses` (§2.1): the views, the sort pose, and one `ModelDraw`
//! per instance with its pose, time, render mode, color transform and overlays.

use ftgsxr_core::{Camera, Pose, UVec2, Vec3};

use super::model::GpuModel;

/// What the shaders need of a view (E6): the camera and the clip mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewParams {
    /// The view's camera in openftgs's conventions (E6), its image W × H included.
    pub camera: Camera,
    /// The size of the target the view draws into, in pixels.
    pub target_size: UVec2,
}

/// E4's eye and E8's forward axis for the shared sort (D-4), in stage space. Per-view
/// sorting takes the eye and the depth from the view's own camera instead (E4, E8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SortPose {
    /// The midpoint of the two eyes (E4's o_v in shared-center mode).
    pub origin: Vec3,
    /// The midpoint pose's forward axis, a unit vector (E8's depth axis).
    pub forward: Vec3,
}

/// How an instance is drawn (§1.4 stages 5 and 6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderMode {
    /// Splats only.
    Splats,
    /// Centers only: discs of `pixel_radius` pixels colored by `colormap`.
    Centers {
        /// ρ, the disc radius in pixels at render scale 1 (default 3, 1–8).
        pixel_radius: f32,
        /// What colors the discs.
        colormap: Colormap,
    },
    /// Splats and centers.
    Both {
        /// ρ, the disc radius in pixels at render scale 1.
        pixel_radius: f32,
        /// What colors the discs.
        colormap: Colormap,
    },
    /// Nothing: the instance is skipped from stage 1 on.
    Hidden,
}

/// What colors an instance's centers (§1.4 stage 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    /// The DC color C₀c₀₀ + 0.5 (the default).
    Dc,
    /// The time center over \[τ₀, τ₁\], through a perceptually ordered colormap.
    TimeCenter,
    /// The speed ‖velocity‖ relative to the model's 99th percentile.
    Speed,
    /// The opacity.
    Opacity,
}

/// E14's transform: gains in linear light; identity = (1, 1, 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTransform {
    /// g·w per channel, applied to the E4 color in linear light and clamped to \[0, 1\].
    pub gain: Vec3,
}

impl ColorTransform {
    /// The identity, (1, 1, 1): the cull pass leaves the E4 color untouched (Q-80).
    pub const IDENTITY: ColorTransform = ColorTransform { gain: Vec3::ONE };
}

/// The debug overlays of one instance (§1.4 Selection and debug views). Declared and
/// carried through `ModelDraw`; the passes do not draw them in M-2 (owner decision on
/// PRD #76).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlays {
    /// The box at the current τ (§1.4), not E10's B_m.
    pub boxes: bool,
    /// Velocity vectors for one primitive in k, when set.
    pub velocities: Option<u32>,
    /// The "time slice" tint by the distance of τ from each time center.
    pub time_tint: bool,
}

/// One instance to draw this frame: a model on the GPU at its pose and time (§1.4 stage 1).
#[derive(Clone, Copy)]
pub struct ModelDraw<'a> {
    /// The model's buffers and temporal index.
    pub model: &'a GpuModel,
    /// P_m, model space → stage space (E5).
    pub pose: Pose,
    /// τ_m, the raw time the instance is evaluated at (E9), in the model's `time_unit`.
    pub tau: f64,
    /// How the instance is drawn.
    pub mode: RenderMode,
    /// E14's color transform.
    pub transform: ColorTransform,
    /// The SH degree to evaluate, at most the loaded degree L′ (a higher value uses L′).
    pub sh_degree: u8,
    /// The debug overlays (not drawn in M-2).
    pub overlays: Overlays,
}
