// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr XR: the OpenXR session, tracking, XR intents, passthrough, depth, light,
//! anchors and the panel (§1.8, §1.9, §2.1). M-3 (#222) brings `XrError`.

#![warn(missing_docs)]
// §2.6: this crate may use `unsafe`, each block with a `// SAFETY:` comment.

mod error;

// §2.1; Q-119: `code` and `user_text`.
pub use error::XrError;
