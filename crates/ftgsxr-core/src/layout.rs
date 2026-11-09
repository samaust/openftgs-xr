// SPDX-License-Identifier: Apache-2.0

//! Packing a model for the GPU (§2.1 `ftgsxr_core::layout`): §1.2's load options 1–3 and
//! 5 applied to the arrays the GPU holds, the static memory estimate of §1.12 that load
//! option 4 compares with the tier's budget, and the opt-in reduction to that budget
//! (D-12, D-13), and E17's trimmed boxes over the packed arrays (D-56).

use half::f16;
use rayon::prelude::*;

use glam::{DVec3, Vec3};

use crate::types::model::{PARALLEL_ROWS, moved_position_f64, percentile_box, visible_interval};
use crate::types::{Aabb, Model, Trim, TrimMode};

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
// Q-22
#[derive(Clone, Debug, PartialEq)]
pub struct GpuArrays {
    /// The number of rows, `order.len()`.
    pub n: u32,
    /// The loaded SH degree L′ = min(L, cap).
    pub sh_degree: u8,
    /// τ₀, the origin of the rebased time centers.
    pub tau0: f64,
    /// τ₁, the end of the model's span (τ₀ for a single timestamp), which gives the
    /// time-center colormap its span \[τ₀, τ₁\] (§1.4 stage 6). // Q-97
    pub tau1: f64,
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
/// τ − τ₀, computed in float64 then rounded, and durations float32 (option 3); τ₀ and τ₁
/// are carried as float64 (Q-97). Packing is parallel over rows (§2.4). `order` may list any rows of `model`, a subset included.
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
        tau1: model.time.tau1, // Q-97
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

/// The GPU memory a model and its frame buffers take (§1.12 Memory, D-13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// The model's static buffers: N times §1.12's bytes per primitive under the options
    /// (the bytes of `pack_for_gpu`'s arrays).
    pub static_bytes: u64,
    /// The frame buffers' bytes per unit of capacity C for one set: records, sort keys and
    /// payloads, histograms. Per-view mode holds one set per view, shared-center mode one.
    pub per_capacity_bytes: u32,
    /// The layer's bytes per pixel of one view.
    pub layer_bytes_per_pixel: u32,
}

/// §1.12's memory table for `n` primitives of degree `sh_degree` loaded with `opts`.
///
/// Static bytes per primitive: position 12, rebased time center 4 and duration 4 always in
/// float32; velocity 12, scale 12, rotation 16, opacity 4 and SH 12 (L′+1)² in float32, half
/// of each for a field `opts.half` names, at the capped degree L′ = min(L, `opts.sh_cap`)
/// (load option 1 comes before the memory check): 64 + 12 (L+1)² in float32 and
/// 42 + 6 (L+1)² in half precision.
///
/// Per unit of capacity, one set of frame buffers: the records, 48 bytes per view or 64
/// bytes when `shared_sort` (shared-center mode, §1.4); the sort keys and payloads, 16
/// bytes (two ping-pong pairs of 32-bit keys and payloads); the histograms, 1 byte. The
/// arguments and statistics, under 1 KiB in all, are not counted. A tier with
/// half-precision shared-center records (`quest-low`'s 48 bytes) takes 16 bytes per unit
/// of capacity less than this estimate, which has no flag for them. The layer: 8 bytes per
/// pixel per view (a 16-bit float RGBA target).
// Q-23
pub fn memory_estimate(
    n: u64,
    sh_degree: u8,
    opts: &GpuLayoutOptions,
    shared_sort: bool,
) -> MemoryEstimate {
    let per_primitive: u64 = field_bytes_per_primitive(sh_degree, opts).iter().sum();
    let record = if shared_sort {
        SHARED_RECORD_BYTES
    } else {
        VIEW_RECORD_BYTES
    };
    MemoryEstimate {
        static_bytes: n * per_primitive,
        per_capacity_bytes: record + SORT_BYTES + HISTOGRAM_BYTES,
        layer_bytes_per_pixel: LAYER_BYTES_PER_PIXEL,
    }
}

/// The number of SH ranges a model's GPU SH array is bound as, at most (Q-376): every
/// other field is bound whole.
// Q-376
pub const SH_BINDINGS: u32 = 8;

/// The largest N whose every field's GPU buffer, padded to whole 32-bit words, fits the
/// device (Q-376, Q-369): the SH buffer in `SH_BINDINGS` SH ranges of W words each, W =
/// ⌊`binding_limit` ÷ 64⌋ · 16 (whole 64-byte offsets); every buffer in `max_buffer`
/// bytes; every other field's buffer in one `binding_limit`-byte binding. The bytes per
/// primitive are §1.12's, at the capped degree L′ = min(L, `opts.sh_cap`).
// Q-376
pub fn binding_row_cap(
    sh_degree: u8,
    opts: &GpuLayoutOptions,
    binding_limit: u64,
    max_buffer: u64,
) -> u64 {
    let sh_ranges = u64::from(SH_BINDINGS) * (binding_limit / 64 * 64);
    let fields = field_bytes_per_primitive(sh_degree, opts);
    fields
        .iter()
        .enumerate()
        .map(|(i, &per_row)| {
            let bound = if i == fields.len() - 1 {
                sh_ranges
            } else {
                binding_limit
            };
            // A buffer of k bytes takes ⌈k ÷ 4⌉ words, so it fits a limit's whole words.
            (bound.min(max_buffer) / 4 * 4) / per_row.max(1)
        })
        .min()
        .unwrap_or(0)
}

/// §1.12's bytes per primitive of each `GpuArrays` field under `opts`, in `GpuArrays`
/// order (position, time center, duration, velocity, scale, rotation, opacity, SH), the
/// SH at L′ = min(L, `opts.sh_cap`).
fn field_bytes_per_primitive(sh_degree: u8, opts: &GpuLayoutOptions) -> [u64; 8] {
    let half = opts.half;
    let bytes = |values: usize, is_half: bool| (values * if is_half { 2 } else { 4 }) as u64;
    [
        bytes(3, false),
        bytes(1, false),
        bytes(1, false),
        bytes(3, half.velocity),
        bytes(3, half.scale),
        bytes(4, half.rotation),
        bytes(1, half.opacity),
        bytes(sh_values(loaded_degree(sh_degree, opts.sh_cap)), half.sh),
    ]
}

/// §1.2 load option 4's opt-in reduction: the `budget` rows of highest importance score
/// opacity·scale_x·scale_y·scale_z, in file order (D-13). The score is computed in
/// float64; at equal scores the earlier file row is kept, so the result is deterministic.
/// A budget of N or more keeps every row.
///
/// # Panics
///
/// If the model has more than `u32::MAX` rows.
// Q-24
pub fn reduce_to_budget(model: &Model, budget: usize) -> Vec<u32> {
    let n = u32::try_from(model.len()).expect("at most u32::MAX rows");
    if budget >= model.len() {
        return (0..n).collect();
    }
    let scores: Vec<f64> = model
        .opacity
        .par_iter()
        .zip(model.scale.par_iter())
        .map(|(&o, s)| f64::from(o) * f64::from(s[0]) * f64::from(s[1]) * f64::from(s[2]))
        .collect();
    // Higher score first, then lower file row: a total order with no equal elements.
    let rank = |a: &u32, b: &u32| {
        scores[*b as usize]
            .total_cmp(&scores[*a as usize])
            .then(a.cmp(b))
    };
    let mut rows: Vec<u32> = (0..n).collect();
    if budget > 0 {
        rows.select_nth_unstable_by(budget - 1, rank);
    }
    rows.truncate(budget);
    rows.sort_unstable();
    rows
}

/// A per-view record's bytes (§1.4 stage 2).
const VIEW_RECORD_BYTES: u32 = 48;
/// A shared-center record's bytes with Σ^S in float32 (§1.4 stage 2).
const SHARED_RECORD_BYTES: u32 = 64;
/// The sort keys and payloads per unit of capacity (§1.12).
const SORT_BYTES: u32 = 16;
/// The histograms per unit of capacity: §1.12's "about C", pinned by its examples (Q-23).
const HISTOGRAM_BYTES: u32 = 1;
/// The layer's bytes per pixel (§1.12).
const LAYER_BYTES_PER_PIXEL: u32 = 8;

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

/// E17's trimmed boxes (B_m, B_c) over the uploaded arrays, in model space; `None` when
/// the trim keeps no row, and when it is off.
///
/// The rows are E10's counted rows, read from `arrays` at their stored precision: a row
/// with opacity σ > `alpha_skip` over its visible interval
/// \[time_center − h, time_center + h\] ∩ \[τ₀, τ₁\], h = duration·√(2 ln(σ/α_skip)), or at
/// τ₀ alone for a single timestamp; with no counted row, every row over \[τ₀, τ₁\]. Each
/// row's kept interval is the part of its interval where μx(τ) (E1) lies inside the trim,
/// one interval or none since μx is linear in τ and both shapes are convex. B_m is the box
/// of the kept rows' positions at their kept intervals' ends, each expanded by 3 · its own
/// largest scale; B_c is E10's percentile box of their positions at the kept intervals'
/// midpoints, or B_m when its longest side is 0. The caller turns `None` into the shape's
/// own box (E17, `Trim::shape_bounds`).
pub fn trimmed_bounds(arrays: &GpuArrays, alpha_skip: f32, trim: &Trim) -> Option<(Aabb, Aabb)> {
    if trim.mode == TrimMode::Off {
        return None;
    }
    let n = arrays.n as usize;
    let span = arrays.tau1 - arrays.tau0; // Q-97: the rebased τ₁
    let counted: Vec<(usize, f64, f64)> = (0..n)
        .filter_map(|k| {
            visible_interval(
                read_value(&arrays.opacity, arrays.half.opacity, k),
                alpha_skip,
                f64::from(arrays.duration[k]),
                f64::from(arrays.time_center[k]),
                (0.0, span),
                arrays.single,
            )
            .map(|(a, b)| (k, a, b))
        })
        .collect();
    let rows = if counted.is_empty() {
        (0..n).map(|k| (k, 0.0, span)).collect()
    } else {
        counted
    };
    // Each kept row: its box at the kept interval's ends with its margin, and its midpoint.
    let one = |&(k, a, b): &(usize, f64, f64)| -> Option<(Vec3, Vec3, Vec3)> {
        let (p, v, tc) = row_motion(arrays, k);
        let (a, b) = kept_interval(trim, p, v, tc, a, b)?;
        let at = |tau: f64| moved_position_f64(p, v, tc, tau);
        let (pa, pb) = (at(a), at(b));
        let margin = Vec3::splat(
            3.0 * Vec3::from_array(read3(&arrays.scale, arrays.half.scale, k)).max_element(),
        );
        Some((pa.min(pb) - margin, pa.max(pb) + margin, at(0.5 * (a + b))))
    };
    let kept: Vec<(Vec3, Vec3, Vec3)> = if rows.len() >= PARALLEL_ROWS {
        rows.par_iter().filter_map(one).collect()
    } else {
        rows.iter().filter_map(one).collect()
    };
    if kept.is_empty() {
        return None;
    }
    let (min, max) = kept.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), &(a, b, _)| (lo.min(a), hi.max(b)),
    );
    let whole = Aabb { min, max };
    let mids: Vec<Vec3> = kept.iter().map(|&(_, _, m)| m).collect();
    let content = percentile_box(&mids)
        .filter(|b| b.size().max_element() > 0.0)
        .unwrap_or(whole);
    Some((whole, content))
}

/// Row `k`'s position, velocity and rebased time center, as E1 takes them.
fn row_motion(arrays: &GpuArrays, k: usize) -> ([f32; 3], [f32; 3], f64) {
    let p = [
        arrays.position[3 * k],
        arrays.position[3 * k + 1],
        arrays.position[3 * k + 2],
    ];
    let v = read3(&arrays.velocity, arrays.half.velocity, k);
    (p, v, f64::from(arrays.time_center[k]))
}

/// The part of \[a, b\] where p + v (τ − tc) lies inside `trim` (box or sphere), in f64;
/// `None` when it is empty.
fn kept_interval(
    trim: &Trim,
    p: [f32; 3],
    v: [f32; 3],
    tc: f64,
    a: f64,
    b: f64,
) -> Option<(f64, f64)> {
    let p = DVec3::from_array(p.map(f64::from));
    let v = DVec3::from_array(v.map(f64::from));
    let (mut lo, mut hi) = (a, b);
    match trim.mode {
        TrimMode::Off => {}
        TrimMode::Box => {
            let (bmin, bmax) = (trim.box_min.as_dvec3(), trim.box_max.as_dvec3());
            for axis in 0..3 {
                if v[axis] == 0.0 {
                    if p[axis] < bmin[axis] || p[axis] > bmax[axis] {
                        return None;
                    }
                } else {
                    // The times τ = tc + (limit − p)/v at which the axis crosses each limit.
                    let t0 = tc + (bmin[axis] - p[axis]) / v[axis];
                    let t1 = tc + (bmax[axis] - p[axis]) / v[axis];
                    lo = lo.max(t0.min(t1));
                    hi = hi.min(t0.max(t1));
                }
            }
        }
        TrimMode::Sphere => {
            // ‖w + v s‖² ≤ r² with w = p − c and s = τ − tc: A s² + B s + C ≤ 0.
            let w = p - trim.center.as_dvec3();
            let r = f64::from(trim.radius);
            let (qa, qb, qc) = (
                v.length_squared(),
                2.0 * v.dot(w),
                w.length_squared() - r * r,
            );
            if qa == 0.0 {
                if qc > 0.0 {
                    return None;
                }
            } else {
                let disc = qb * qb - 4.0 * qa * qc;
                if disc < 0.0 {
                    return None;
                }
                let root = disc.sqrt();
                lo = lo.max(tc + (-qb - root) / (2.0 * qa));
                hi = hi.min(tc + (-qb + root) / (2.0 * qa));
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Value `m` of a `GpuArrays` byte array, float32 or binary16 (`GpuArrays`' layout).
fn read_value(bytes: &[u8], half: bool, m: usize) -> f32 {
    if half {
        f16::from_bits(u16::from_le_bytes([bytes[2 * m], bytes[2 * m + 1]])).to_f32()
    } else {
        f32::from_le_bytes([
            bytes[4 * m],
            bytes[4 * m + 1],
            bytes[4 * m + 2],
            bytes[4 * m + 3],
        ])
    }
}

/// Row `k`'s three values of a `GpuArrays` byte array of width 3.
fn read3(bytes: &[u8], half: bool, k: usize) -> [f32; 3] {
    std::array::from_fn(|i| read_value(bytes, half, 3 * k + i))
}
