// SPDX-License-Identifier: Apache-2.0

//! The model and the small types every other module shares (§2.1 `ftgsxr_core::types`):
//! the exchange model in memory and its reader rules (§1.2), its time span and render
//! conventions, the camera of E6, the pose of E5, the box of E10 and the error catalog
//! of §1.15.

mod camera;
mod conventions;
mod error;
mod model;
mod pose;
mod time;

pub use camera::{Camera, Fov, Intrinsics};
pub use conventions::RenderConventions;
pub use error::CoreError;
pub use model::{Aabb, Model, ModelInfo, ModelKind};
pub use pose::Pose;
pub use time::TimeSpan;
