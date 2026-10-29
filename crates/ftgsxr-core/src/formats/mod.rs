// SPDX-License-Identifier: Apache-2.0

//! openftgs's request and renders formats and the `info` reply (§2.1 `ftgsxr_core::formats`,
//! §1.7, openftgs §2.2): the `ftgs-render-request` reader, the `ftgs-renders` writer and
//! reader, the `ftgs-scene` manifest reader and the adapter reply of openftgs §3.3.
//!
//! The readers apply openftgs's own checks (its `io.read_render_request`, `io.read_renders`
//! and `io.read_scene_manifest`), each refusal under the §1.15 code of Q-T14.1.

mod json;
mod request;

pub use request::{RenderItem, RenderRequest, read_request};
