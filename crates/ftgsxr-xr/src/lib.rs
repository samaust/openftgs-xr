// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr XR: the OpenXR session, tracking, XR intents, passthrough, depth, light,
//! anchors and the panel (§1.8, §1.9, §2.1). M-3 brings `XrError` (#222), the session,
//! `XrPlugin` and the `Environment` declarations (#223).

#![warn(missing_docs)]
// §2.6: this crate may use `unsafe`, each block with a `// SAFETY:` comment.

mod anchors;
mod depth;
mod display;
mod error;
mod light;
mod panel;
mod passthrough;
mod plugin;
mod session;
mod tracking;
mod xr_input;

// §2.1 `AnchorId`, `AnchorStore`, `NoAnchors` (§1.9, D-26, #571; Q-343).
pub use anchors::{AnchorId, AnchorStore, NoAnchors};
// §2.1 `EnvironmentDepth` (§1.9, E13, #570; Q-349).
pub use depth::EnvironmentDepth;
// §2.1 (#228, Q-124).
pub use display::DisplaySettings;
// Q-133: `DisplaySettings.foveation`'s type, for `ftgsxr-quest`, whose band (§2.1)
// excludes `ftgsxr-render`.
pub use ftgsxr_render::Foveation;
// §2.1; Q-119: `code` and `user_text`.
pub use error::XrError;
// §2.1 `LightingMode`, `LightEstimate`, `light_gains`, `manual_gains`, `grey_world_gains`
// (E14, D-38, #666); Q-382: `LightEstimate::from_thumb`, `band_mask`, `LightSmoother`,
// `LightingMode::from_settings`.
pub use light::{
    LightEstimate, LightSmoother, LightingMode, band_mask, grey_world_gains, light_gains,
    manual_gains,
};
// §2.1 `MainPanel`, `PanelFollow` (#462); Q-265, Q-305: the placement functions.
pub use panel::{MainPanel, PanelFollow, summon_pose, wrist_menu_pose, wrist_menu_visible};
// §2.1 (M-3: the declarations I-15 reads).
pub use passthrough::{Environment, EnvironmentMode, PassthroughStyle};
// Q-342: the contact shadows' plugin (§1.9, #569).
pub use passthrough::ContactShadowPlugin;
// §2.1 `XrPlugin`; Q-120: `add_xr`; Q-220: `runtime_unavailable`.
pub use plugin::{XrPlugin, add_xr, runtime_unavailable};
// §2.1 (#227).
pub use tracking::{HandSource, TrackedHand, XrHands};
// §2.1 (#460); Q-302: `XrInputState::set_selected`.
pub use xr_input::{XrInputConfig, XrInputState, hands_to_intents};
