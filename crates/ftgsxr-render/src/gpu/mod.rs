// SPDX-License-Identifier: Apache-2.0

//! The wgpu render layer (§2.1 `ftgsxr_render::gpu`), usable without a Bevy `App`: tiers,
//! device, `GpuModel`, frame buffers, the frame inputs, the passes (`SplatPasses`: so far
//! the cull and size passes of stages 1–3 and the statistics of stage 8) and the frame
//! buffers' read back; the sort, the draws and the headless renderer join in the later
//! M-2 tickets.

mod device;
mod error;
mod frame;
mod inputs;
mod model;
mod passes;
mod readback;
mod stats;
mod tier;

pub use device::{BudgetSource, MemoryBudget, SplatDevice}; // Q-68, Q-74
pub use error::RenderError;
pub use frame::FrameBuffers;
pub use inputs::{ColorTransform, Colormap, ModelDraw, Overlays, RenderMode, SortPose, ViewParams};
pub use model::GpuModel;
pub use passes::SplatPasses;
pub use readback::{FrameContents, RecordView}; // Q-69
pub use stats::{FrameStats, GpuTimes};
pub use tier::{Capacity, Foveation, QualityTier, SortKey, SortMode};
