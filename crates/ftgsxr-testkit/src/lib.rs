// SPDX-License-Identifier: Apache-2.0

//! openftgs-xr test kit: fixtures and their checksums, image metrics (E15), synthetic
//! tiny models, the `gpu` lane's device and its validation layer's messages (Q-105),
//! the in-process fixture server (§2.7, Q-1) and per-process scratch directories (Q-62).
//! A dev-dependency only.

#![warn(missing_docs)]

mod exchange;
mod fixtures;
mod gpu;
mod layer_log;
mod metrics;
mod probes;
mod random;
mod real_sh;
mod safetensors;
mod scratch;
mod server;

pub use exchange::exchange_bytes;
pub use fixtures::fixtures_dir;
pub use gpu::gpu_context; // Q-1
pub use layer_log::{LayerMessages, layer_messages}; // Q-105
pub use metrics::{Metrics, metrics};
pub use probes::{probe, reference_render};
pub use random::random_model;
pub use real_sh::real_sh;
pub use safetensors::SafetensorsBuilder;
pub use scratch::{process_dir, scratch_dir}; // Q-62
pub use server::{LocalServer, RequestRecord, ServedListing, serve_fixtures};
