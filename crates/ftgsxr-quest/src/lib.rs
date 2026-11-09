// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr on the Meta Quest 3: the Android entry point and the JNI adapter of the
//! shim (§2.5, §2.1), built as `libftgsxr_quest.so`. On the host the crate is empty
//! (§2.7's host checks), its tests aside; `main` is Android's (M-3, #224).

#![warn(missing_docs)]
// §2.6: this crate may use `unsafe` for JNI, each block with a `// SAFETY:` comment.

#[cfg(target_os = "android")]
mod android;
#[cfg(any(test, target_os = "android"))]
mod camera;
#[cfg(target_os = "android")]
mod device_test;
#[cfg(any(test, target_os = "android"))]
mod dv2_rate;
#[cfg(any(test, target_os = "android"))]
mod exit;
#[cfg(target_os = "android")]
mod jni_calls;
#[cfg(target_os = "android")]
mod platform;
#[cfg(target_os = "android")]
mod shim;
#[cfg(any(test, target_os = "android"))]
mod tree;

// §2.1 `ftgsxr_quest::QuestPlatform`.
#[cfg(target_os = "android")]
pub use platform::QuestPlatform;
