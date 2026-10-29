// SPDX-License-Identifier: Apache-2.0

//! The exchange model `ftgs-if` v1 (§1.2, openftgs §2.2, openftgs `io.read_exchange`) and its
//! header-only information (§1.5).

use std::collections::HashMap;

use safetensors::tensor::{Dtype, Metadata};

use super::container::{self, Container, check_tensor, f32_le, f32s_le, f64_le, values};
use super::{SAFETENSORS_PREFIX, check_version, normalize_rotations, sha256};
use crate::types::{CoreError, Model, ModelInfo, ModelKind, RenderConventions, TimeSpan};

/// The `format` of an exchange model.
const FORMAT: &str = "ftgs-if";

/// What the header of an exchange model tells, checked (§1.2 reader rules up to the
/// tensors' dtypes and shapes). The conventions block's own fault is held back, since
/// §1.2's order reports it after every rule on the arrays (PRD #1, "Reader internals").
struct Header {
    n: usize,
    sh_degree: u8,
    time: TimeSpan,
    producer: String,
    conventions: Result<RenderConventions, CoreError>,
}

/// The metadata entry `key`, or E-FILE-KEYS.
fn required<'a>(meta: &'a HashMap<String, String>, key: &str) -> Result<&'a str, CoreError> {
    meta.get(key)
        .map(String::as_str)
        .ok_or_else(|| CoreError::Keys(key.to_owned()))
}

/// `format` and `version` (§1.2): a missing key is E-FILE-KEYS, another `format`
/// E-FILE-FORMAT, a version that is not `major[.minor]` of major 1 E-FILE-VERSION (Q-31).
// Q-31
pub(crate) fn check_format(meta: &HashMap<String, String>, format: &str) -> Result<(), CoreError> {
    let found = required(meta, "format")?;
    if found != format {
        return Err(CoreError::Format(format!(
            "format {found:?}, expected {format:?}"
        )));
    }
    check_version(required(meta, "version")?)
}

/// Checks the header of an exchange model in §1.2's order: format, version, the metadata
/// keys and numbers, then the eight tensors present (E-FILE-KEYS) with their dtype and shape
/// (E-FILE-SHAPE; the `sh` coefficient count E-FILE-SH).
fn check_header(header: &Metadata) -> Result<Header, CoreError> {
    let meta = container::metadata(header);
    check_format(&meta, FORMAT)?;
    for key in ["sh_degree", "tau0", "tau1", "time_unit"] {
        required(&meta, key)?;
    }
    let degree = required(&meta, "sh_degree")?;
    let sh_degree: i64 = degree
        .parse()
        .map_err(|_| CoreError::Value(format!("sh_degree = {degree:?}")))?;
    let sh_degree = u8::try_from(sh_degree)
        .ok()
        .filter(|l| *l <= 3)
        .ok_or_else(|| CoreError::Sh(format!("sh_degree {sh_degree}")))?;
    let number = |key: &str| -> Result<f64, CoreError> {
        let text = required(&meta, key)?;
        text.parse::<f64>()
            .map_err(|_| CoreError::Value(format!("{key} = {text:?}")))
    };
    let (tau0, tau1) = (number("tau0")?, number("tau1")?);
    if !tau0.is_finite() || !tau1.is_finite() || tau1 < tau0 {
        return Err(CoreError::Value(format!("tau0 = {tau0}, tau1 = {tau1}")));
    }
    let time = TimeSpan {
        tau0,
        tau1,
        time_unit: required(&meta, "time_unit")?.to_owned(),
    };

    let keys = [
        "position",
        "time_center",
        "duration",
        "velocity",
        "scale",
        "rotation",
        "opacity",
        "sh",
    ];
    for key in keys {
        if header.info(key).is_none() {
            return Err(CoreError::Keys(key.to_owned()));
        }
    }
    let info = |key: &str| header.info(key).expect("invariant: presence checked above");
    let position = info("position");
    let n = match position.shape[..] {
        [n, 3] if n >= 1 => n,
        _ => {
            return Err(CoreError::Shape(format!(
                "position has shape {:?}, expected [N, 3] with N ≥ 1",
                position.shape
            )));
        }
    };
    check_tensor("position", position, Dtype::F32, &[n, 3])?;
    check_tensor("time_center", info("time_center"), Dtype::F64, &[n])?;
    check_tensor("duration", info("duration"), Dtype::F64, &[n])?;
    check_tensor("velocity", info("velocity"), Dtype::F32, &[n, 3])?;
    check_tensor("scale", info("scale"), Dtype::F32, &[n, 3])?;
    check_tensor("rotation", info("rotation"), Dtype::F32, &[n, 4])?;
    check_tensor("opacity", info("opacity"), Dtype::F32, &[n])?;
    let sh = info("sh");
    let k = (usize::from(sh_degree) + 1).pow(2);
    match sh.shape[..] {
        [rows, kk, 3] if rows == n && kk != k => {
            return Err(CoreError::Sh(format!(
                "sh has shape {:?}, expected [{n}, {k}, 3] for sh_degree {sh_degree}",
                sh.shape
            )));
        }
        _ => check_tensor("sh", sh, Dtype::F32, &[n, k, 3])?,
    }

    let conventions = match meta.get("render_conventions") {
        None => Ok(RenderConventions::default()),
        Some(json) => RenderConventions::from_json(json),
    };
    Ok(Header {
        n,
        sh_degree,
        time,
        producer: meta.get("producer").cloned().unwrap_or_default(),
        conventions,
    })
}

/// The header-only information of an exchange model (§1.5), from the JSON header alone:
/// `header` is the bytes that follow the 8-byte prefix (`header_len` of them), `file_bytes`
/// the whole file's size. Every rule of §1.2 a header can show is checked, in
/// `read_exchange`'s order: the container (the header's offsets must cover the file's
/// `file_bytes` exactly), format, version, the metadata keys and numbers, the tensors'
/// presence, dtypes and shapes, and the `render_conventions` block. The values themselves
/// are not seen, so a file that passes may still be refused by `read_exchange`. Only an
/// exchange model has this read (Q-33).
// Q-33
pub fn read_info(header: &[u8], file_bytes: u64) -> Result<ModelInfo, CoreError> {
    let parsed = container::parse_header(header)?;
    let total = (SAFETENSORS_PREFIX + header.len() + parsed.data_len()) as u64;
    if total != file_bytes {
        return Err(CoreError::Format(format!(
            "a file of {file_bytes} bytes where the header gives {total}"
        )));
    }
    let h = check_header(&parsed)?;
    Ok(ModelInfo {
        n: h.n as u64,
        sh_degree: h.sh_degree,
        time: h.time,
        producer: h.producer,
        conventions: h.conventions?,
        kind: ModelKind::ExchangeFile,
        file_bytes,
    })
}

/// Reads an exchange model `ftgs-if` v1 (§1.2): the safetensors container, its header and
/// tensors, then `Model::validate`, reporting the first rule broken in §1.2's order with its
/// E-FILE code; unknown tensors and metadata keys are ignored. The rotations are then
/// normalized with the canonical sign (Q-32), and `identity` is the SHA-256 of `bytes`,
/// computed beside the decoding (D-40).
pub fn read_exchange(bytes: &[u8]) -> Result<Model, CoreError> {
    let (identity, model) = rayon::join(|| sha256(bytes), || decode(bytes));
    let mut model = model?;
    model.identity = identity;
    Ok(model)
}

/// The container and header checks, the arrays, `Model::validate`, the conventions block.
fn decode(bytes: &[u8]) -> Result<Model, CoreError> {
    let file = Container::parse(bytes)?;
    let h = check_header(file.header())?;
    let tensor = |key: &str| {
        file.bytes(
            file.header()
                .info(key)
                .expect("invariant: presence checked by check_header"),
        )
    };
    let mut model = Model {
        position: values(tensor("position"), f32s_le::<3>),
        time_center: values(tensor("time_center"), f64_le),
        duration: values(tensor("duration"), f64_le),
        velocity: values(tensor("velocity"), f32s_le::<3>),
        scale: values(tensor("scale"), f32s_le::<3>),
        rotation: values(tensor("rotation"), f32s_le::<4>),
        opacity: values(tensor("opacity"), f32_le),
        sh: values(tensor("sh"), f32_le),
        sh_degree: h.sh_degree,
        time: h.time,
        conventions: RenderConventions::default(),
        producer: h.producer,
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    };
    debug_assert_eq!(model.len(), h.n);
    finish(&mut model, h.conventions)?;
    Ok(model)
}

/// Validates a decoded model and normalizes its rotations. §1.2's order reports every rule
/// on the arrays before a fault of the conventions block (Q-8), so a block that did not
/// parse is reported only once the arrays pass (validated meanwhile with the defaults).
pub(crate) fn finish(
    model: &mut Model,
    conventions: Result<RenderConventions, CoreError>,
) -> Result<(), CoreError> {
    match conventions {
        Ok(c) => {
            model.conventions = c;
            model.validate()?;
        }
        Err(e) => {
            model.validate()?;
            return Err(e);
        }
    }
    normalize_rotations(model);
    Ok(())
}
