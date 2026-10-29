// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr foundation: the model and its three readers (§1.2), E1–E5 and E8–E11
//! on the CPU, the temporal index, the clock, the GPU layout, and the openftgs request
//! and renders formats (§2.1). M-1 fills its modules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod covariance;
pub mod types;

pub use types::{
    Aabb, Camera, CoreError, Fov, Intrinsics, Model, ModelInfo, ModelKind, Pose, RenderConventions,
    TimeSpan,
};

pub use covariance::{covariance, posed_covariance, quat_from_wxyz, quat_to_wxyz};

/// glam's types, re-exported from the workspace pin so every crate uses the engine's glam (§2.1, §2.3).
pub use glam::{DMat3, DMat4, DVec3, Mat3, Mat4, Quat, UVec2, Vec2, Vec3};
