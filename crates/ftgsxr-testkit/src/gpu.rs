// SPDX-License-Identifier: Apache-2.0

//! The `gpu` lane's device (§2.7 Organisation, Q-1).

use ftgsxr_render::SplatDevice;

/// A `SplatDevice` of its own for a `gpu` test (§2.7, Q-1): `SplatDevice::headless`'s
/// Vulkan instance, adapter and device, with the validation layers and the shader
/// validator on in debug builds and the validation collector installed, so the test fails
/// on any message of wgpu's own validation (§1.13, R-16). The Vulkan layer's messages go
/// to the log instead: a test watches them with `layer_messages`, made first (Q-T121.1).
///
/// # Panics
///
/// When no Vulkan adapter is found (E-GPU-ADAPTER): the `gpu` lane needs one.
pub fn gpu_context() -> SplatDevice {
    SplatDevice::headless().expect("the gpu lane needs a Vulkan adapter (E-GPU-ADAPTER)")
}
