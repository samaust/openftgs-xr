// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr test kit: fixtures and their checksums, image metrics (E15), synthetic
//! tiny models and the in-process fixture server (§2.7, Q-1). A dev-dependency only.

#![warn(missing_docs)]

mod exchange;
mod fixtures;
mod random;
mod real_sh;
mod safetensors;

pub use exchange::exchange_bytes;
pub use fixtures::fixtures_dir;
pub use random::random_model;
pub use real_sh::real_sh;
pub use safetensors::SafetensorsBuilder;
