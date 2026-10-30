// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr engine: the wgpu render layer and its Bevy plugin adapter (§1.4, §2.1).
//! M-2 fills it; the `bevy` module joins with the plugin adapter.

#![warn(missing_docs)]

pub mod gpu;

pub use gpu::{
    BudgetSource, Capacity, ColorTransform, Colormap, Foveation, FrameBuffers, FrameContents,
    FrameStats, GpuModel, GpuTimes, HeadlessRenderer, MemoryBudget, ModelDraw, ModelId,
    OcclusionInput, Overlays, QualityTier, RecordView, RenderError, RenderMode, SortKey, SortMode,
    SortPose, SplatDevice, SplatPasses, ViewParams,
};
