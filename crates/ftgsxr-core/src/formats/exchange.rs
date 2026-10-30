// SPDX-License-Identifier: Apache-2.0

//! The `ftgs-if` v1 writer of `import` and `export` (§1.2, §1.7, openftgs §2.2, openftgs
//! `io.write_exchange`; Q-65, Q-T109.1).

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::types::{CoreError, Model};

/// The data section's alignment: the header is padded with spaces so the data starts at a
/// multiple of 8 bytes in the file, and every float64 tensor comes first, so every tensor
/// is aligned to its dtype.
const ALIGN: usize = 8;

/// The model as `ftgs-if` v1 bytes (§1.2), the arrays written as they are, unchecked, and
/// the producer as `model.producer` gives it. The header is compact JSON: `__metadata__`
/// first, with `format` = `ftgs-if`, `version` = `1`, `sh_degree`, `time_unit`, `tau0` and
/// `tau1` as shortest round-trip decimals, `producer` (written even when empty) and
/// `render_conventions` = `RenderConventions::to_json`, in that order; then the eight
/// tensors of §1.2's table, float64 first, then by name, their data in the same order. The
/// bytes depend on the model alone; `identity` and `kind` are not part of the file.
// Q-T109.1
pub fn encode_exchange(model: &Model) -> Vec<u8> {
    encode(model, &model.producer)
}

/// Writes the model as a `ftgs-if` v1 file at `path` (§1.2, §1.7), as openftgs's
/// `io.write_exchange` writes one: the model is validated first (`Model::validate`, its
/// E-FILE code), then written as `encode_exchange` gives it, with the producer
/// `openftgs-xr <version>` in place of `model.producer`. The file is written to a temporary
/// sibling `.<name>.tmp-<pid>` renamed over `path`, so a failed write leaves the old file,
/// or none, and no temporary; an I/O failure is `CoreError::Io` (E-IO).
// Q-65, Q-T109.1
pub fn write_exchange(path: &Path, model: &Model) -> Result<(), CoreError> {
    model.validate()?;
    let producer = format!("openftgs-xr {}", env!("CARGO_PKG_VERSION"));
    let bytes = encode(model, &producer);
    let name = path
        .file_name()
        .ok_or_else(|| {
            CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} names no file", path.display()),
            ))
        })?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let written = fs::File::create(&temporary)
        .and_then(|mut f| f.write_all(&bytes).and_then(|()| f.sync_all()))
        .and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    Ok(written?)
}

/// `encode_exchange` with the given producer.
fn encode(model: &Model, producer: &str) -> Vec<u8> {
    let n = model.position.len();
    let k = (usize::from(model.sh_degree) + 1).pow(2);
    let t = &model.time;
    let metadata = [
        ("format", "ftgs-if".to_owned()),
        ("version", "1".to_owned()),
        ("sh_degree", model.sh_degree.to_string()),
        ("time_unit", t.time_unit.clone()),
        ("tau0", format!("{:?}", t.tau0)),
        ("tau1", format!("{:?}", t.tau1)),
        ("producer", producer.to_owned()),
        ("render_conventions", model.conventions.to_json()),
    ];
    let tensors: [(&str, &str, Vec<usize>, Vec<u8>); 8] = [
        ("duration", "F64", vec![n], f64_bytes(&model.duration)),
        ("time_center", "F64", vec![n], f64_bytes(&model.time_center)),
        ("opacity", "F32", vec![n], f32_bytes(&model.opacity)),
        (
            "position",
            "F32",
            vec![n, 3],
            f32_bytes(model.position.as_flattened()),
        ),
        (
            "rotation",
            "F32",
            vec![n, 4],
            f32_bytes(model.rotation.as_flattened()),
        ),
        (
            "scale",
            "F32",
            vec![n, 3],
            f32_bytes(model.scale.as_flattened()),
        ),
        ("sh", "F32", vec![n, k, 3], f32_bytes(&model.sh)),
        (
            "velocity",
            "F32",
            vec![n, 3],
            f32_bytes(model.velocity.as_flattened()),
        ),
    ];

    let mut header = String::from("{\"__metadata__\":{");
    for (i, (key, value)) in metadata.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        header.push_str(&json_string(key));
        header.push(':');
        header.push_str(&json_string(value));
    }
    header.push('}');
    let mut offset = 0usize;
    for (name, dtype, shape, bytes) in &tensors {
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let end = offset + bytes.len();
        header.push_str(&format!(
            ",{}:{{\"dtype\":{},\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            json_string(name),
            json_string(dtype),
            shape.join(","),
        ));
        offset = end;
    }
    header.push('}');
    let start = 8 + header.len();
    header.push_str(&" ".repeat((ALIGN - start % ALIGN) % ALIGN));

    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, bytes) in &tensors {
        out.extend_from_slice(bytes);
    }
    out
}

/// `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    serde_json::to_string(s).expect("invariant: a string serializes to JSON")
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
