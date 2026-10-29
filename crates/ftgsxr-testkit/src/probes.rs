// SPDX-License-Identifier: Apache-2.0

//! The committed probes and their reference renders (§2.7 Fixtures, Q-1, Q-T15.3).

use std::path::PathBuf;

use ftgsxr_core::{Model, RenderImage, RenderRequest, open_model, read_renders, read_request};

use crate::fixtures_dir;

/// The committed probe `name` (README Layout: `probes/NAME.safetensors` and
/// `NAME.request.json`): its model, read by `open_model`, and its render request, read by
/// `read_request` (Q-T15.3).
///
/// # Panics
///
/// When either file is missing or does not read, naming it.
pub fn probe(name: &str) -> (Model, RenderRequest) {
    let model_path = probe_file(name, "safetensors");
    let model =
        open_model(&model_path).unwrap_or_else(|e| panic!("probe {}: {e}", model_path.display()));
    (model, request(name))
}

/// The reference renders of the committed probe `name` (`probes/NAME.renders.safetensors`,
/// openftgs's reference backend), sorted by item id, after checking that they were made from
/// the committed request (its SHA-256, `read_renders`; Q-T15.3).
///
/// # Panics
///
/// When the renders or the request are missing or do not read, or the digests differ.
pub fn reference_render(name: &str) -> Vec<(String, RenderImage)> {
    let request = request(name);
    let path = probe_file(name, "renders.safetensors");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("probe {}: {e}", path.display()));
    read_renders(&bytes, Some(request.sha256))
        .unwrap_or_else(|e| panic!("probe {}: {e}", path.display()))
}

/// The probe's render request.
fn request(name: &str) -> RenderRequest {
    let path = probe_file(name, "request.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("probe {}: {e}", path.display()));
    read_request(&bytes).unwrap_or_else(|e| panic!("probe {}: {e}", path.display()))
}

/// `probes/<name>.<extension>` under the verified fixtures directory.
fn probe_file(name: &str, extension: &str) -> PathBuf {
    fixtures_dir().join(format!("probes/{name}.{extension}"))
}
