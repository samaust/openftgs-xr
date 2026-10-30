// SPDX-License-Identifier: Apache-2.0

//! A `Model` written as an exchange file `ftgs-if` v1 (§1.2, Q-1), without the loader.

use ftgsxr_core::Model;

/// The model as a `ftgs-if` v1 file (§1.2): the eight tensors of §1.2's table with their
/// dtypes and shapes, and the metadata openftgs's `io.write_exchange` writes (source log
/// 30): `format` = `ftgs-if`, `version` = `1`, `sh_degree`, `time_unit`, `tau0` and
/// `tau1` as shortest round-trip decimals (Rust's `{:?}`, Python's `repr` for ordinary
/// values), `producer` (written even when empty) and `render_conventions` =
/// `RenderConventions::to_json`. The arrays are written as they are, unchecked; `identity`
/// and `kind` are not part of the file. The tensors' data is ordered 8-byte dtypes first,
/// then by name, so every tensor is aligned in the file.
///
/// It is `ftgsxr_core::encode_exchange`, the encoder of the player's own exchange writer,
/// so one writer serves the command line and the tests (Q-65, Q-72).
// Q-65, Q-72
pub fn exchange_bytes(model: &Model) -> Vec<u8> {
    ftgsxr_core::encode_exchange(model)
}
