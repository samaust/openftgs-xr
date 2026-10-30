// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr engine: the wgpu render layer and its Bevy plugin adapter (§1.4, §2.1).
//! `gpu` runs without a Bevy `App`; `bevy` is the plugin adapter that mounts it in one.

#![warn(missing_docs)]

pub mod bevy;
pub mod gpu;

pub use crate::bevy::{
    DepthFrame, OutputSpace, SplatAsset, SplatCamera, SplatEvent, SplatLoad, SplatMode, SplatModel,
    SplatOcclusion, SplatOverlays, SplatPlugin, SplatStats, SplatTierState, SplatTime,
    SplatTransform, load_splat_asset,
};
pub use gpu::{
    BudgetSource, Capacity, ColorTransform, Colormap, Foveation, FrameBuffers, FrameContents,
    FrameStats, GpuModel, GpuTimes, HeadlessRenderer, MemoryBudget, ModelDraw, ModelId,
    OcclusionInput, Overlays, QualityTier, RecordView, RenderError, RenderMode, SortKey, SortMode,
    SortPose, SplatDevice, SplatPasses, ViewParams,
};
