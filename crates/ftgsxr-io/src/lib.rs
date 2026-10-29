// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr foundation: paths, settings, scene and pose files, listings, HTTP and
//! model sources (§1.5, §1.11, §2.1). M-1 fills its modules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
pub mod paths;

pub use error::IoError;
pub use paths::AppPaths;
