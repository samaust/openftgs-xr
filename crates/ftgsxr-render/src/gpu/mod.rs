// SPDX-License-Identifier: Apache-2.0

//! The wgpu render layer (§2.1 `ftgsxr_render::gpu`), usable without a Bevy `App`: tiers,
//! device, `GpuModel`, frame buffers; the passes, shaders, statistics and the headless
//! renderer join in the later M-2 tickets.

mod device;
mod error;
mod frame;
mod model;
mod tier;

pub use device::{BudgetSource, MemoryBudget, SplatDevice}; // Q-68, Q-74
pub use error::RenderError;
pub use frame::FrameBuffers;
pub use model::GpuModel;
pub use tier::{Capacity, Foveation, QualityTier, SortKey, SortMode};
