// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr application: everything the desktop and Quest builds share (§1.5–§1.7,
//! §1.10–§1.13, §2.1). M-2 adds the headless commands that need no GPU (#117).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod headless;

pub use headless::{HeadlessError, cmd_info, exit_code};
