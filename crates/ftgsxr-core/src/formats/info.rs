// SPDX-License-Identifier: Apache-2.0

//! The `info` reply (§1.7): the adapter reply of openftgs §3.3's contract, as openftgs's
//! `ftgs-adapter info` writes it.

use crate::types::RenderConventions;

/// The reply of `ftgsxr info` (§1.7, openftgs §3.3).
#[derive(Clone, Debug, PartialEq)]
pub struct InfoReply {
    /// The implementation's name, `openftgs-xr`.
    pub name: String,
    /// The player's version.
    pub version: String,
    /// The adapter queries answered; none for the player.
    pub query_kinds: Vec<String>,
    /// The rasterizer's name, `openftgs-xr wgpu-blend`.
    pub rasterizer_name: String,
    /// The rasterizer's version.
    pub rasterizer_version: String,
    /// The conventions the player renders with: openftgs's, with `transmittance_stop`
    /// `None` since E7 applies none (D-7).
    pub render_conventions: RenderConventions,
}

/// The reply as one compact JSON object with openftgs's keys in openftgs's order: `name`,
/// `version`, `query_kinds`, `rasterizer` {`name`, `version`} and `render_conventions`
/// (`RenderConventions::to_json`, so `transmittance_stop` is `null` when `None`). The reply
/// is written as given; the caller sets the stop to `None` (D-7).
pub fn info_json(reply: &InfoReply) -> String {
    fn text(s: &str) -> String {
        serde_json::to_string(s).expect("invariant: a string serializes to JSON")
    }
    let kinds: Vec<String> = reply.query_kinds.iter().map(|k| text(k)).collect();
    format!(
        "{{\"name\":{},\"version\":{},\"query_kinds\":[{}],\"rasterizer\":{{\"name\":{},\"version\":{}}},\"render_conventions\":{}}}",
        text(&reply.name),
        text(&reply.version),
        kinds.join(","),
        text(&reply.rasterizer_name),
        text(&reply.rasterizer_version),
        reply.render_conventions.to_json(),
    )
}
