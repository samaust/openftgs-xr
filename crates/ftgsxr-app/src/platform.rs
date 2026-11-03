// SPDX-License-Identifier: Apache-2.0

//! The platform port (§2.1 `ftgsxr_app::platform`, §2.5): the file pickers, the runtime
//! permissions, the documents and the passthrough camera, behind one trait with two
//! adapters, `ftgsxr_desktop::LinuxPlatform` and `ftgsxr_quest::QuestPlatform`.

use std::sync::Arc;
use std::task::Poll;

use bevy::prelude::Resource;
use ftgsxr_core::Camera;
use ftgsxr_io::{AppPaths, DocumentProvider, ModelRef};

/// What a picker picks (§1.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PickKind {
    /// One or more model files.
    Files,
    /// A folder, added as a source.
    Folder,
}

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

/// A pending pick, polled with `Platform::poll_pick`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PickHandle(pub u32);

/// A pending permission request, polled with `Platform::poll_permission`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PermissionHandle(pub u32);

/// A passthrough camera frame reduced by the shim (§1.9, D-15): 64 × 48 RGB in [0, 1],
/// row-major, with the camera's pose and field of view in stage space when known.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraThumb {
    /// The width in pixels.
    pub width: u32,
    /// The height in pixels.
    pub height: u32,
    /// `width × height × 3` values, row-major.
    pub rgb: Vec<f32>,
    /// When the frame was taken, in seconds.
    pub at_s: f64,
    /// The camera, when known.
    pub camera: Option<Camera>,
}

/// The platform port: two adapters, `LinuxPlatform` (rfd dialogs, every permission
/// granted, no documents, no camera) and `QuestPlatform` (the JNI shim). Non-blocking:
/// the caller polls.
pub trait Platform: Send + Sync {
    /// The player's directories (§1.11).
    fn paths(&self) -> &AppPaths;
    /// Opens a picker; its result is polled with `poll_pick`.
    fn pick(&self, kind: PickKind) -> PickHandle;
    /// The picked models (empty when cancelled), or the reason the picker failed;
    /// `Pending` while it is open.
    fn poll_pick(&self, h: PickHandle) -> Poll<Result<Vec<ModelRef>, String>>;
    /// Requests a permission; its answer is polled with `poll_permission`.
    fn request_permission(&self, p: Permission) -> PermissionHandle;
    /// Whether the permission was granted; `Pending` until the user answers.
    fn poll_permission(&self, h: PermissionHandle) -> Poll<bool>;
    /// The documents of picked trees (D-49), where the platform has them.
    fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>>;
    /// The passthrough camera's frames (§1.9), where the platform has them.
    fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>>;
}

/// The passthrough camera's reduced frames (§1.9, D-15).
pub trait CameraFrames: Send + Sync {
    /// Starts the frames, after the permissions of §1.9 are granted.
    fn start(&self) -> Result<(), String>;
    /// The newest thumbnail, `None` until the first arrives.
    fn latest(&self) -> Option<CameraThumb>;
    /// Stops the frames.
    fn stop(&self);
}

/// The platform in the world.
#[derive(Resource, Clone)]
pub struct PlatformRes(pub Arc<dyn Platform>);
