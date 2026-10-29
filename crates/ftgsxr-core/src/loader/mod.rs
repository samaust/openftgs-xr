// SPDX-License-Identifier: Apache-2.0

//! The three readers of §1.2 (the exchange model `ftgs-if`, the checkpoint
//! `ftgs-checkpoint`, the PLY snapshot), the header-only read of §1.5, validation and the
//! model identity (D-40, D-41).
//!
//! Every reader checks its container and header first, then builds a `Model` and runs
//! `Model::validate` on it, so the first rule broken is reported in §1.2's order: format,
//! version, metadata keys and numbers, tensors present with their dtype and shape, then
//! the in-memory rules, and the `render_conventions` block last. The identity digest runs
//! beside the decoding (`rayon::join`), so the bytes are read once by each.

mod checkpoint;
mod container;
mod exchange;
mod ply;

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::types::{CoreError, Model};

pub use checkpoint::read_checkpoint;
pub use exchange::{read_exchange, read_info};
pub use ply::read_ply;

/// The bytes before a safetensors header: its length as a little-endian u64 (§1.2).
pub const SAFETENSORS_PREFIX: usize = 8;

/// The header length a safetensors prefix gives: its 8 bytes as a little-endian u64 (§1.2).
/// A header-only read (§1.5) fetches `SAFETENSORS_PREFIX` + this many bytes and passes the
/// header to `read_info`.
pub fn header_len(prefix: &[u8; 8]) -> u64 {
    u64::from_le_bytes(*prefix)
}

/// The SHA-256 of `bytes`: a model's identity (D-40).
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// An identity as 64 lowercase hexadecimal digits, as `sha256sum` prints it (D-40).
pub fn hex(identity: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in identity {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    out
}

/// Opens a model by its kind (§1.2): a directory holding `manifest.json` is a checkpoint
/// (`read_checkpoint`), a file with a `.ply` suffix (any case) a PLY snapshot (`read_ply`),
/// any other file an exchange model (`read_exchange`). The file is read into memory, not
/// memory-mapped: `ftgsxr-core` forbids `unsafe` (§2.6) and mapping a file is `unsafe`
/// (Q-30). A directory without `manifest.json` is E-FILE-FORMAT; a file that cannot be
/// read is E-IO, its message naming the path.
// Q-30
pub fn open_model(path: &Path) -> Result<Model, CoreError> {
    if path.is_dir() {
        if !path.join(checkpoint::MANIFEST).is_file() {
            return Err(CoreError::Format(format!(
                "{} is a directory without {}",
                path.display(),
                checkpoint::MANIFEST
            )));
        }
        return read_checkpoint(path);
    }
    let bytes = read_file(path)?;
    let is_ply = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("ply"));
    if is_ply {
        read_ply(&bytes)
    } else {
        read_exchange(&bytes)
    }
}

/// The file's bytes, or E-IO naming the path.
pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>, CoreError> {
    std::fs::read(path).map_err(|e| {
        CoreError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })
}

/// Rows above which the readers convert arrays in parallel (PRD #1, "Reader internals").
pub(crate) const PARALLEL_ROWS: usize = 1 << 16;

/// `f` over every element, in parallel when `parallel` (the caller compares the row count
/// with `PARALLEL_ROWS`).
pub(crate) fn map_rows<T: Sync, U: Send>(
    parallel: bool,
    xs: &[T],
    f: impl Fn(&T) -> U + Sync + Send,
) -> Vec<U> {
    use rayon::prelude::*;
    if parallel {
        xs.par_iter().map(f).collect()
    } else {
        xs.iter().map(f).collect()
    }
}

/// The bound below which a quaternion's norm counts as 1 and the quaternion is kept as
/// written: a few float32 rounding steps of a writer's normalization (Q-32).
const UNIT_TOLERANCE: f64 = 1e-6;

/// A rotation as the `Model` holds it (§1.2): normalized and with the canonical sign
/// (w ≥ 0; where w = 0 the first nonzero component positive, §1.14). The norm is taken in
/// float64; a quaternion already unit within `UNIT_TOLERANCE` keeps its components, others
/// are divided by the norm in float64 and rounded to float32 (Q-32). The caller has
/// checked the norm (`Model::validate`: > 10⁻⁸).
// Q-32
pub(crate) fn canonical_unit(q: [f32; 4]) -> [f32; 4] {
    let wide = q.map(f64::from);
    let norm = wide.iter().map(|x| x * x).sum::<f64>().sqrt();
    let unit = if (norm - 1.0).abs() <= UNIT_TOLERANCE {
        q
    } else {
        wide.map(|x| (x / norm) as f32)
    };
    let negative = unit.iter().find(|c| **c != 0.0).is_some_and(|c| *c < 0.0);
    // + 0.0 turns the −0.0 a sign flip makes of a zero component into +0.0.
    if negative {
        unit.map(|c| -c + 0.0)
    } else {
        unit.map(|c| c + 0.0)
    }
}

/// Normalizes every rotation of a validated model (§1.2: readers accept non-unit
/// quaternions and normalize them), in parallel above `PARALLEL_ROWS`.
pub(crate) fn normalize_rotations(model: &mut Model) {
    use rayon::prelude::*;
    if model.rotation.len() >= PARALLEL_ROWS {
        model
            .rotation
            .par_iter_mut()
            .for_each(|q| *q = canonical_unit(*q));
    } else {
        model
            .rotation
            .iter_mut()
            .for_each(|q| *q = canonical_unit(*q));
    }
}

/// A `version` text `major[.minor]` in decimal without leading zeros, of major 1
/// (§1.2: another major is rejected, a minor part ignored); anything else is
/// E-FILE-VERSION (Q-31). The model readers and `formats`' readers (openftgs
/// `io._check_header`, openftgs §2.2) share it.
// Q-31
pub(crate) fn check_version(text: &str) -> Result<(), CoreError> {
    fn decimal(part: &str) -> Option<u64> {
        let digits = !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
        let leading_zero = part.len() > 1 && part.starts_with('0');
        (digits && !leading_zero)
            .then(|| part.parse().ok())
            .flatten()
    }
    let mut parts = text.splitn(2, '.');
    let major = parts.next().and_then(decimal);
    let minor_ok = parts.next().is_none_or(|m| decimal(m).is_some());
    match major {
        Some(1) if minor_ok => Ok(()),
        _ => Err(CoreError::Version(format!("{text:?}"))),
    }
}

/// openftgs's activation of an opacity logit: sigmoid in float64, clamped to
/// \[10⁻⁷, 1 − 10⁻⁷\], rounded to float32 (§1.2, openftgs §1.1).
pub(crate) fn opacity_from_logit(logit: f32) -> f32 {
    const EPS: f64 = 1e-7;
    let x = f64::from(logit);
    let s = 1.0 / (1.0 + (-x).exp());
    s.clamp(EPS, 1.0 - EPS) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_unit_keeps_unit_quaternions_and_flips_signs() {
        assert_eq!(canonical_unit([1.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            canonical_unit([-0.5, 0.5, -0.5, 0.5]),
            [0.5, -0.5, 0.5, -0.5]
        );
        let q = canonical_unit([0.0, -0.0, -2.0, 0.0]);
        assert_eq!(q, [0.0, 0.0, 1.0, 0.0]);
        assert!(q.iter().all(|c| c.is_sign_positive()));
        assert_eq!(canonical_unit([0.0, 0.0, 0.0, 1e-9]), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn versions_of_major_one_only() {
        for ok in ["1", "1.0", "1.3", "1.12"] {
            assert!(check_version(ok).is_ok(), "{ok}");
        }
        for bad in ["2", "0", "01", "1.", ".1", "1.01", "1.2.3", "", "one", " 1"] {
            assert_eq!(
                check_version(bad).unwrap_err().code(),
                "E-FILE-VERSION",
                "{bad}"
            );
        }
    }

    #[test]
    fn opacity_logit_clamps() {
        assert_eq!(opacity_from_logit(0.0), 0.5);
        assert_eq!(opacity_from_logit(100.0), (1.0f64 - 1e-7) as f32);
        assert_eq!(opacity_from_logit(-100.0), 1e-7f64 as f32);
    }
}
