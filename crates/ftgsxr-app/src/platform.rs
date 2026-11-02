// SPDX-License-Identifier: Apache-2.0

//! The platform port (§2.1 `ftgsxr_app::platform`). M-3 brings `Permission`, which
//! `ftgsxr_xr::XrError` carries; the port itself and its other types are M-4's.

/// A runtime permission a feature needs (§1.9, §1.15 E-XR-PERMISSION).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Permission {
    /// The passthrough cameras.
    Camera,
    /// The scene model (planes, meshes, depth).
    Scene,
    /// Hand tracking.
    HandTracking,
    /// Spatial anchors.
    Anchors,
}
