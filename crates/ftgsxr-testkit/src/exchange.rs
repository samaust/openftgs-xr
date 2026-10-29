// SPDX-License-Identifier: Apache-2.0

//! A `Model` written as an exchange file `ftgs-if` v1 (§1.2, Q-1), without the loader.

use ftgsxr_core::Model;

use crate::SafetensorsBuilder;

/// The model as a `ftgs-if` v1 file (§1.2): the eight tensors of §1.2's table with their
/// dtypes and shapes, and the metadata openftgs's `io.write_exchange` writes (source log
/// 30): `format` = `ftgs-if`, `version` = `1`, `sh_degree`, `time_unit`, `tau0` and
/// `tau1` as shortest round-trip decimals (Rust's `{:?}`, Python's `repr` for ordinary
/// values), `producer` (written even when empty) and `render_conventions` =
/// `RenderConventions::to_json`. The arrays are written as they are, unchecked; `identity`
/// and `kind` are not part of the file. The tensors' data is ordered 8-byte dtypes first,
/// then by name, so every tensor is aligned in the file.
pub fn exchange_bytes(model: &Model) -> Vec<u8> {
    let n = model.position.len();
    let k = (usize::from(model.sh_degree) + 1).pow(2);
    let t = &model.time;
    SafetensorsBuilder::new()
        .metadata("format", "ftgs-if")
        .metadata("version", "1")
        .metadata("sh_degree", &model.sh_degree.to_string())
        .metadata("time_unit", &t.time_unit)
        .metadata("tau0", &format!("{:?}", t.tau0))
        .metadata("tau1", &format!("{:?}", t.tau1))
        .metadata("producer", &model.producer)
        .metadata("render_conventions", &model.conventions.to_json())
        .tensor("duration", "F64", &[n], f64_bytes(&model.duration))
        .tensor("time_center", "F64", &[n], f64_bytes(&model.time_center))
        .tensor("opacity", "F32", &[n], f32_bytes(&model.opacity))
        .tensor(
            "position",
            "F32",
            &[n, 3],
            f32_bytes(bytemuck::cast_slice(&model.position)),
        )
        .tensor(
            "rotation",
            "F32",
            &[n, 4],
            f32_bytes(bytemuck::cast_slice(&model.rotation)),
        )
        .tensor(
            "scale",
            "F32",
            &[n, 3],
            f32_bytes(bytemuck::cast_slice(&model.scale)),
        )
        .tensor("sh", "F32", &[n, k, 3], f32_bytes(&model.sh))
        .tensor(
            "velocity",
            "F32",
            &[n, 3],
            f32_bytes(bytemuck::cast_slice(&model.velocity)),
        )
        .build()
}

/// Float32 values as little-endian bytes (§1.2).
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    if cfg!(target_endian = "little") {
        bytemuck::cast_slice(values).to_vec()
    } else {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

/// Float64 values as little-endian bytes (§1.2).
fn f64_bytes(values: &[f64]) -> Vec<u8> {
    if cfg!(target_endian = "little") {
        bytemuck::cast_slice(values).to_vec()
    } else {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}
