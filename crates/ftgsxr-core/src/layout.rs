// SPDX-License-Identifier: Apache-2.0

//! Packing a model for the GPU (§2.1 `ftgsxr_core::layout`): §1.2's load options 1–3 and
//! 5 applied to the arrays the GPU holds, the static memory estimate of §1.12 that load
//! option 4 compares with the tier's budget, and the opt-in reduction to that budget
//! (D-12, D-13).

use half::f16;
use rayon::prelude::*;

use crate::types::Model;

/// Which fields are stored on the GPU as IEEE half floats (§1.2 load option 2, D-12).
/// Position, the rebased time center and the duration are always float32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HalfFields {
    /// `velocity` in half precision.
    pub velocity: bool,
    /// `scale` in half precision.
    pub scale: bool,
    /// `rotation` in half precision.
    pub rotation: bool,
    /// `opacity` in half precision.
    pub opacity: bool,
    /// `sh` in half precision.
    pub sh: bool,
}

/// The load options that shape the GPU arrays (§1.2 load options 1 and 2): the storage
/// precision of each field and the SH cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuLayoutOptions {
    /// The fields stored in half precision.
    pub half: HalfFields,
    /// The SH cap L′: bands above it are dropped; `None` or a cap at or above L keeps L.
    pub sh_cap: Option<u8>,
}

/// The flat arrays the GPU holds, rows permuted by `order`, time rebased to τ − τ₀
/// (§1.2 load options).
///
/// Structure of arrays, one array per field, row-major with no padding (§2.4): row k of
/// every array is file row `order[k]`, and a row's values follow the `Model` field's order
/// (rotation (w, x, y, z); SH coefficient-major, RGB innermost, (L′+1)² coefficients).
///
/// - `position`, `time_center` and `duration` are always float32 values.
/// - `velocity`, `scale`, `rotation`, `opacity` and `sh` are bytes. When the field's flag
///   in `half` is false they hold little-endian IEEE float32 values, 4 bytes each. When it
///   is true they hold little-endian IEEE binary16 values, 2 bytes each, rounded to nearest
///   with ties to even; the GPU reads them as `u32` pairs, value 2j in bits 0–15 of word j
///   and value 2j + 1 in bits 16–31 (WGSL's `unpack2x16float`), so value m of the array
///   is in word m/2 whatever row it belongs to.
///
/// Each byte array is exactly the size of its values, so a half array with an odd number
/// of values ends mid-word; padding it to whole `u32` words for the buffer is the
/// uploader's (`memory_estimate` counts the values, §1.12).
// Q-T13.1
#[derive(Clone, Debug, PartialEq)]
pub struct GpuArrays {
    /// The number of rows, `order.len()`.
    pub n: u32,
    /// The loaded SH degree L′ = min(L, cap).
    pub sh_degree: u8,
    /// τ₀, the origin of the rebased time centers.
    pub tau0: f64,
    /// Whether the model has a single timestamp (D-9).
    pub single: bool,
    /// Which of the byte arrays hold half floats.
    pub half: HalfFields,
    /// \[n, 3\] μx.
    pub position: Vec<f32>,
    /// \[n\] time_center − τ₀.
    pub time_center: Vec<f32>,
    /// \[n\] duration.
    pub duration: Vec<f32>,
    /// \[n, 3\] velocity.
    pub velocity: Vec<u8>,
    /// \[n, 3\] scale.
    pub scale: Vec<u8>,
    /// \[n, 4\] rotation (w, x, y, z).
    pub rotation: Vec<u8>,
    /// \[n\] opacity.
    pub opacity: Vec<u8>,
    /// \[n, (L′+1)², 3\] SH coefficients.
    pub sh: Vec<u8>,
}

/// Packs `model` for the GPU: row k of every array is file row `order[k]` (new row →
/// file row, the direction `TemporalIndex::build` returns, §1.2 load option 5); the SH
/// bands above `opts.sh_cap` are dropped (option 1); the fields `opts.half` names are
/// rounded to half floats, to nearest even (option 2, D-12); time centers become float32
/// τ − τ₀, computed in float64 then rounded, and durations float32 (option 3). Packing is
/// parallel over rows (§2.4). `order` may list any rows of `model`, a subset included.
///
/// # Panics
///
/// If an entry of `order` is not a row of `model`, or `order` has more than `u32::MAX`
/// entries.
pub fn pack_for_gpu(model: &Model, order: &[u32], opts: &GpuLayoutOptions) -> GpuArrays {
    let n = u32::try_from(order.len()).expect("at most u32::MAX rows");
    let degree = loaded_degree(model.sh_degree, opts.sh_cap);
    let width = sh_values(model.sh_degree);
    let keep = sh_values(degree);
    let tau0 = model.time.tau0;
    let half = opts.half;
    GpuArrays {
        n,
        sh_degree: degree,
        tau0,
        single: model.is_single_timestamp(),
        half,
        position: gather_f32(model.position.as_flattened(), 3, order),
        time_center: order
            .par_iter()
            .map(|&i| (model.time_center[i as usize] - tau0) as f32)
            .collect(),
        duration: order
            .par_iter()
            .map(|&i| model.duration[i as usize] as f32)
            .collect(),
        velocity: gather_bytes(model.velocity.as_flattened(), 3, 3, order, half.velocity),
        scale: gather_bytes(model.scale.as_flattened(), 3, 3, order, half.scale),
        rotation: gather_bytes(model.rotation.as_flattened(), 4, 4, order, half.rotation),
        opacity: gather_bytes(&model.opacity, 1, 1, order, half.opacity),
        sh: gather_bytes(&model.sh, width, keep, order, half.sh),
    }
}

/// L′ = min(L, cap), the degree the GPU holds (§1.2 load option 1).
fn loaded_degree(sh_degree: u8, sh_cap: Option<u8>) -> u8 {
    sh_cap.map_or(sh_degree, |cap| cap.min(sh_degree))
}

/// The SH values per row at a degree, (L+1)²·3.
fn sh_values(degree: u8) -> usize {
    (usize::from(degree) + 1).pow(2) * 3
}

/// The rows `order` names of a float32 array of `width` values per row.
fn gather_f32(values: &[f32], width: usize, order: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; order.len() * width];
    out.par_chunks_mut(width)
        .zip(order.par_iter())
        .for_each(|(dst, &i)| {
            let i = i as usize;
            dst.copy_from_slice(&values[i * width..(i + 1) * width]);
        });
    out
}

/// The first `keep` values of the rows `order` names of an array of `width` values per
/// row, as little-endian float32 bytes or, when `half`, little-endian binary16 bytes
/// rounded to nearest even (`GpuArrays`' byte layout).
fn gather_bytes(values: &[f32], width: usize, keep: usize, order: &[u32], half: bool) -> Vec<u8> {
    let size = if half { 2 } else { 4 };
    let mut out = vec![0u8; order.len() * keep * size];
    out.par_chunks_mut(keep * size)
        .zip(order.par_iter())
        .for_each(|(dst, &i)| {
            let start = i as usize * width;
            let src = &values[start..start + keep];
            for (d, &x) in dst.chunks_exact_mut(size).zip(src) {
                if half {
                    d.copy_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
                } else {
                    d.copy_from_slice(&x.to_le_bytes());
                }
            }
        });
    out
}
