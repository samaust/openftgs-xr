// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr XR: the OpenXR session, tracking, XR intents, passthrough, depth, light,
//! anchors and the panel (§1.8, §1.9, §2.1). M-3 brings `XrError` (#222), the session,
//! `XrPlugin` and the `Environment` declarations (#223).

#![warn(missing_docs)]
// §2.6: this crate may use `unsafe`, each block with a `// SAFETY:` comment.

mod display;
mod error;
mod panel;
mod passthrough;
mod plugin;
mod session;
mod tracking;
mod xr_input;

// §2.1 (#228, Q-124).
pub use display::DisplaySettings;
// Q-133: `DisplaySettings.foveation`'s type, for `ftgsxr-quest`, whose band (§2.1)
// excludes `ftgsxr-render`.
pub use ftgsxr_render::Foveation;
// §2.1; Q-119: `code` and `user_text`.
pub use error::XrError;
// §2.1 `MainPanel`, `PanelFollow` (#462); Q-265, Q-T462.1: the placement functions.
pub use panel::{MainPanel, PanelFollow, summon_pose, wrist_menu_pose, wrist_menu_visible};
// §2.1 (M-3: the declarations I-15 reads).
pub use passthrough::{Environment, EnvironmentMode, PassthroughStyle};
// §2.1 `XrPlugin`; Q-120: `add_xr`; Q-220: `runtime_unavailable`.
pub use plugin::{XrPlugin, add_xr, runtime_unavailable};
// §2.1 (#227).
pub use tracking::{HandSource, TrackedHand, XrHands};
// §2.1 (#460); Q-302: `XrInputState::set_selected`.
pub use xr_input::{XrInputConfig, XrInputState, hands_to_intents};
