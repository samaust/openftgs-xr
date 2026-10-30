// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr foundation: the model and its three readers (§1.2), E1–E5 and E8–E11
//! on the CPU, the temporal index, the clock, the GPU layout, and the openftgs request
//! and renders formats (§2.1). M-1 fills its modules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod clock;
pub mod covariance;
pub mod formats;
pub mod index;
pub mod layout;
pub mod loader;
pub mod motion;
pub mod pose;
pub mod sh;
pub mod types;

// Q-65, Q-T109.1: the exchange writer.
pub use formats::{
    InfoReply, ManifestImage, RenderImage, RenderItem, RenderRequest, SceneManifest,
    encode_exchange, info_json, read_renders, read_request, read_scene_manifest, write_exchange,
    write_renders,
};
pub use pose::{Grab, Preset, ScaleLimits, TwoHandGrab, fit_pose, ray_hit, snap_to_floor};
pub use types::{
    Aabb, Camera, CoreError, Fov, Intrinsics, Model, ModelInfo, ModelKind, Pose, RenderConventions,
    TimeSpan,
};

pub use sh::{SH_C0, sh_basis, sh_color};

pub use covariance::{
    Projected, alpha_at, covariance, occlusion_weight, posed_covariance, project, quat_from_wxyz,
    quat_to_wxyz, sort_key, sort_key16,
};
pub use motion::{effective_opacity, moved_position, temporal_opacity, temporal_window};

pub use index::{DispatchRange, TemporalIndex};

pub use clock::{Clock, EndRule, LoopMode, ModelTime, PlayState};
pub use layout::{
    GpuArrays, GpuLayoutOptions, HalfFields, MemoryEstimate, memory_estimate, pack_for_gpu,
    reduce_to_budget,
};
pub use loader::{
    SAFETENSORS_PREFIX, header_len, hex, open_model, read_checkpoint, read_exchange, read_info,
    read_ply, sha256,
};

/// glam's types, re-exported from the workspace pin so every crate uses the engine's glam (§2.1, §2.3).
pub use glam::{DMat3, DMat4, DVec3, Mat3, Mat4, Quat, UVec2, Vec2, Vec3};
