// SPDX-License-Identifier: Apache-2.0

//! The wgpu render layer (§2.1 `ftgsxr_render::gpu`), usable without a Bevy `App`: tiers,
//! device, `GpuModel`, frame buffers, the frame inputs, the passes (`SplatPasses`: the cull
//! and size passes of stages 1–3, the radix sort of stage 4, the splat and centers draws
//! of stages 5 and 6, the resolve of stage 7 and the statistics of stage 8), the frame
//! buffers' read back and the headless renderer.

mod device;
mod error;
mod frame;
mod headless;
mod inputs;
mod model;
mod passes;
mod readback;
mod sort;
mod stats;
mod tier;

pub use device::{BudgetSource, MemoryBudget, SplatDevice}; // Q-68, Q-74
pub use error::RenderError;
pub use frame::FrameBuffers;
pub use headless::{HeadlessRenderer, ModelId};
pub use inputs::{
    ColorTransform, Colormap, ModelDraw, OcclusionInput, Overlays, RenderMode, SortPose, ViewParams,
};
pub use model::GpuModel;
pub use passes::SplatPasses;
pub use readback::{FrameContents, RecordView}; // Q-69
pub use stats::{FrameStats, GpuTimes};
pub use tier::{Capacity, Foveation, QualityTier, SortKey, SortMode};
