// SPDX-License-Identifier: Apache-2.0

//! The wgpu render layer (§2.1 `ftgsxr_render::gpu`), usable without a Bevy `App`: tiers,
//! device, `GpuModel`, frame buffers, the frame inputs, the passes (`SplatPasses`: the cull
//! and size passes of stages 1–3, the radix sort of stage 4, the splat and centers draws
//! of stages 5 and 6, the resolve of stage 7 and the statistics of stage 8), the frame
//! buffers' read back, the layer probe, the headless renderer and the import of a Vulkan image (Q-339).

pub(crate) mod device;
mod error;
pub(crate) mod frame;
pub(crate) mod headless;
mod import;
mod inputs;
pub(crate) mod model;
pub(crate) mod passes;
pub(crate) mod probe;
mod readback;
mod sort;
mod stats;
pub(crate) mod tier;

pub use device::{BudgetSource, MemoryBudget, SplatDevice}; // Q-68, Q-74
pub use error::RenderError;
pub use frame::FrameBuffers;
pub use headless::{HeadlessRenderer, ModelId};
pub use import::import_vulkan_texture; // Q-339
pub use inputs::{
    ColorTransform, Colormap, ModelDraw, OcclusionInput, Overlays, RenderMode, SortPose, ViewParams,
};
pub use model::GpuModel;
pub use passes::SplatPasses;
pub use probe::LayerSums;
pub use readback::{FrameContents, RecordView, SharedRecordView}; // Q-69, Q-400
pub use stats::{FrameStats, GpuTimes};
pub use tier::{Capacity, Foveation, QualityTier, SortKey, SortMode};
