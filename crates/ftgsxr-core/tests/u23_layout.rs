// SPDX-License-Identifier: Apache-2.0

//! U-23 (`cpu` lane): `ftgsxr_core::layout`, §1.2's load options and §1.12's memory
//! table. The permutation applied; time rebased; half fields within 2⁻¹¹ relative and
//! rounded to nearest even; position and time unchanged; `memory_estimate` against
//! §1.12's table; `reduce_to_budget` keeps the top scores in file order, deterministic.

use ftgsxr_core::layout::{
    GpuArrays, GpuLayoutOptions, HalfFields, MemoryEstimate, memory_estimate, pack_for_gpu,
    reduce_to_budget,
};
use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};

/// A valid degree-1 model of `n` rows whose every value names its row, so a permuted row
/// is recognizable: row i has position (i, 10 + i, 20 + i), time center τ₀ + 0.5 + i, …
fn model(n: usize, tau0: f64, tau1: f64) -> Model {
    let f = |i: usize| i as f32;
    Model {
        position: (0..n).map(|i| [f(i), 10.0 + f(i), 20.0 + f(i)]).collect(),
        time_center: (0..n).map(|i| tau0 + 0.5 + i as f64).collect(),
        duration: (0..n).map(|i| 0.25 + 0.125 * i as f64).collect(),
        velocity: (0..n).map(|i| [0.5 + f(i), -1.0 - f(i), 0.25]).collect(),
        scale: (0..n).map(|i| [0.01 * (1.0 + f(i)), 0.02, 0.03]).collect(),
        rotation: (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    [1.0, 0.0, 0.0, 0.0]
                } else {
                    [0.0, 0.6, 0.8, 0.0]
                }
            })
            .collect(),
        opacity: (0..n).map(|i| 0.1 + 0.2 * (i % 4) as f32).collect(),
        sh: (0..n)
            .flat_map(|i| (0..12).map(move |k| 0.25 * k as f32 - f(i)))
            .collect(),
        sh_degree: 1,
        time: TimeSpan {
            tau0,
            tau1,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

const ALL_HALF: HalfFields = HalfFields {
    velocity: true,
    scale: true,
    rotation: true,
    opacity: true,
    sh: true,
};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

/// One IEEE binary16 value from its bits, written from the format's definition (sign,
/// 5-bit exponent of bias 15, 10-bit fraction; subnormals at exponent 0), sharing no code
/// with the packer.
fn binary16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff) / 1024.0;
    let magnitude = match exponent {
        0 => fraction * 2f64.powi(-14),
        31 => {
            if fraction == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + fraction) * 2f64.powi(e - 15),
    };
    (sign * magnitude) as f32
}

/// Decodes little-endian IEEE binary16 values.
fn halves(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| binary16(u16::from_le_bytes(b)))
        .collect()
}

/// A field of the packed arrays as f32 values, decoded by its storage precision.
fn decode(bytes: &[u8], is_half: bool) -> Vec<f32> {
    if is_half { halves(bytes) } else { f32s(bytes) }
}

/// The per-row fields of `arrays` row k, decoded, in the model's units (time rebased).
fn packed_rows(a: &GpuArrays) -> Vec<Vec<f32>> {
    let k = (usize::from(a.sh_degree) + 1).pow(2) * 3;
    let (v, s, r, o, sh) = (
        decode(&a.velocity, a.half.velocity),
        decode(&a.scale, a.half.scale),
        decode(&a.rotation, a.half.rotation),
        decode(&a.opacity, a.half.opacity),
        decode(&a.sh, a.half.sh),
    );
    (0..a.n as usize)
        .map(|i| {
            let mut row = a.position[3 * i..3 * i + 3].to_vec();
            row.extend([a.time_center[i], a.duration[i]]);
            row.extend(&v[3 * i..3 * i + 3]);
            row.extend(&s[3 * i..3 * i + 3]);
            row.extend(&r[4 * i..4 * i + 4]);
            row.push(o[i]);
            row.extend(&sh[k * i..k * i + k]);
            row
        })
        .collect()
}

/// U-23: row k of the packed arrays is file row `order[k]`, in every field and both
/// precisions.
#[test]
fn u23_permutation_applied() {
    let m = model(5, 0.0, 8.0);
    let order = [3u32, 0, 4, 2, 1];
    let opts32 = GpuLayoutOptions::default();
    let identity: Vec<u32> = (0..5).collect();
    for half in [HalfFields::default(), ALL_HALF] {
        let opts = GpuLayoutOptions { half, sh_cap: None };
        let base = packed_rows(&pack_for_gpu(&m, &identity, &opts));
        let a = pack_for_gpu(&m, &order, &opts);
        assert_eq!(a.n, 5);
        assert_eq!(a.half, half);
        let rows = packed_rows(&a);
        for (k, &i) in order.iter().enumerate() {
            assert_eq!(rows[k], base[i as usize], "row {k} ← file row {i}");
        }
    }
    // With float32 storage the identity order reproduces the file's rows exactly.
    let a = pack_for_gpu(&m, &identity, &opts32);
    for (i, row) in packed_rows(&a).iter().enumerate() {
        assert_eq!(row[0..3], m.position[i]);
        assert_eq!(row[5..8], m.velocity[i]);
        assert_eq!(row[8..11], m.scale[i]);
        assert_eq!(row[11..15], m.rotation[i]);
        assert_eq!(row[15], m.opacity[i]);
        assert_eq!(row[16..28], m.sh[12 * i..12 * i + 12]);
    }
}

/// U-23: time centers rebased to float32 τ − τ₀ and durations float32, at τ₀ = 10⁹ s
/// where the raw centers lose their fraction in float32; position and time stay float32
/// and unchanged whatever fields are half (D-12).
#[test]
fn u23_time_rebased_position_and_time_unchanged() {
    let tau0 = 1.0e9;
    let m = model(4, tau0, tau0 + 8.0);
    let order = [2u32, 3, 0, 1];
    for half in [HalfFields::default(), ALL_HALF] {
        let a = pack_for_gpu(&m, &order, &GpuLayoutOptions { half, sh_cap: None });
        assert_eq!(a.tau0, tau0);
        assert!(!a.single);
        for (k, &i) in order.iter().enumerate() {
            let i = i as usize;
            // model(): time center τ₀ + 0.5 + i, duration 0.25 + 0.125 i, both exact.
            assert_eq!(a.time_center[k], 0.5 + i as f32);
            assert_eq!(a.duration[k], 0.25 + 0.125 * i as f32);
            assert_eq!(a.position[3 * k..3 * k + 3], m.position[i]);
        }
    }
}

/// U-23: a single-timestamp model (D-9) packs with `single` set, every rebased center 0.
#[test]
fn u23_single_timestamp_rebases_to_zero() {
    let mut m = model(3, 5.0, 5.0);
    m.time_center = vec![5.0; 3];
    m.duration = vec![1.0; 3];
    m.velocity = vec![[0.0; 3]; 3];
    m.validate().unwrap();
    let a = pack_for_gpu(&m, &[0, 1, 2], &GpuLayoutOptions::default());
    assert!(a.single);
    assert_eq!(a.tau0, 5.0);
    assert_eq!(a.time_center, [0.0; 3]);
    assert_eq!(a.duration, [1.0; 3]);
}

/// A deterministic generator of values in [lo, hi) (xorshift64*), for test inputs only.
struct Values(u64);
impl Values {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let u = (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
}

/// U-23: every half field decodes within 2⁻¹¹ relative of the float32 value for values of
/// order 1 (§1.12's half-precision bounds), on 2,000 rows of degree 3.
#[test]
fn u23_half_fields_within_2_pow_minus_11_relative() {
    let n = 2000;
    let mut g = Values(0x9e37_79b9_7f4a_7c15);
    let mut m = model(n, 0.0, 8.0);
    let signed = |g: &mut Values| {
        let x = g.next(1.0e-3, 4.0);
        if g.next(0.0, 1.0) < 0.5 { -x } else { x }
    };
    m.velocity = (0..n)
        .map(|_| [signed(&mut g), signed(&mut g), signed(&mut g)])
        .collect();
    m.scale = (0..n)
        .map(|_| [g.next(1e-3, 2.0), g.next(1e-3, 2.0), g.next(1e-3, 2.0)])
        .collect();
    m.rotation = (0..n)
        .map(|_| {
            [
                g.next(1e-3, 1.0),
                signed(&mut g),
                signed(&mut g),
                signed(&mut g),
            ]
        })
        .collect();
    m.opacity = (0..n).map(|_| g.next(1e-3, 0.999)).collect();
    m.sh_degree = 3;
    m.sh = (0..n * 48).map(|_| signed(&mut g)).collect();
    let order: Vec<u32> = (0..n as u32).rev().collect();
    let a = pack_for_gpu(
        &m,
        &order,
        &GpuLayoutOptions {
            half: ALL_HALF,
            sh_cap: None,
        },
    );
    let bound = 2f32.powi(-11);
    let check = |name: &str, packed: &[u8], file: &[f32], width: usize| {
        let decoded = halves(packed);
        assert_eq!(decoded.len(), n * width, "{name}");
        for (k, &i) in order.iter().enumerate() {
            let row = &file[i as usize * width..(i as usize + 1) * width];
            for (j, (&h, &x)) in decoded[k * width..(k + 1) * width]
                .iter()
                .zip(row)
                .enumerate()
            {
                let rel = ((h - x) / x).abs();
                assert!(rel <= bound, "{name}[{i}][{j}]: {x} → {h}, relative {rel}");
            }
        }
    };
    check("velocity", &a.velocity, m.velocity.as_flattened(), 3);
    check("scale", &a.scale, m.scale.as_flattened(), 3);
    check("rotation", &a.rotation, m.rotation.as_flattened(), 4);
    check("opacity", &a.opacity, &m.opacity, 1);
    check("sh", &a.sh, &m.sh, 48);
}

/// U-23: half fields round to nearest with ties to even (§1.2 load option 2), which
/// truncation or ties away from zero would not: 1 + 2⁻¹¹ lies halfway between 1 and
/// 1 + 2⁻¹⁰ and goes to 1 (even); 1 + 3·2⁻¹¹ lies halfway between 1 + 2⁻¹⁰ and 1 + 2⁻⁹ and
/// goes to 1 + 2⁻⁹ (even); 1 + 2⁻¹¹ + 2⁻¹⁵ lies past the half and goes up to 1 + 2⁻¹⁰.
#[test]
fn u23_half_rounds_to_nearest_even() {
    let cases: [(f32, f32); 3] = [
        (1.0 + 2f32.powi(-11), 1.0),
        (1.0 + 3.0 * 2f32.powi(-11), 1.0 + 2f32.powi(-9)),
        (1.0 + 2f32.powi(-11) + 2f32.powi(-15), 1.0 + 2f32.powi(-10)),
    ];
    let mut m = model(3, 0.0, 8.0);
    for (i, &(x, _)) in cases.iter().enumerate() {
        m.velocity[i] = [x, -x, x];
        m.scale[i] = [x; 3];
        m.sh[12 * i..12 * i + 12].fill(x);
    }
    let a = pack_for_gpu(
        &m,
        &[0, 1, 2],
        &GpuLayoutOptions {
            half: ALL_HALF,
            sh_cap: None,
        },
    );
    let (v, s, sh) = (halves(&a.velocity), halves(&a.scale), halves(&a.sh));
    for (i, &(x, want)) in cases.iter().enumerate() {
        assert_eq!(v[3 * i..3 * i + 3], [want, -want, want], "velocity {x}");
        assert_eq!(s[3 * i..3 * i + 3], [want; 3], "scale {x}");
        assert_eq!(sh[12 * i..12 * i + 12], [want; 12], "sh {x}");
    }
}

/// §1.2 load option 1 in the packer: a cap below L keeps each row's first (L′+1)²
/// coefficients and sets the degree to L′; a cap at or above L is the identity.
#[test]
fn u23_sh_cap_applied() {
    let m = model(3, 0.0, 8.0);
    let order = [1u32, 2, 0];
    let capped = pack_for_gpu(
        &m,
        &order,
        &GpuLayoutOptions {
            sh_cap: Some(0),
            ..Default::default()
        },
    );
    assert_eq!(capped.sh_degree, 0);
    let sh = f32s(&capped.sh);
    assert_eq!(sh.len(), 3 * 3);
    for (k, &i) in order.iter().enumerate() {
        let i = i as usize;
        // model(): row i's coefficient k is 0.25 k − i; the DC term is k = 0, 1, 2.
        assert_eq!(
            sh[3 * k..3 * k + 3],
            [-(i as f32), 0.25 - i as f32, 0.5 - i as f32]
        );
    }
    for cap in [None, Some(1), Some(3)] {
        let a = pack_for_gpu(
            &m,
            &order,
            &GpuLayoutOptions {
                sh_cap: cap,
                ..Default::default()
            },
        );
        assert_eq!(a.sh_degree, 1, "{cap:?}");
        assert_eq!(f32s(&a.sh).len(), 3 * 12, "{cap:?}");
    }
}

/// U-23: `memory_estimate` equals §1.12's table of static bytes per primitive for
/// L = 0 … 3: float32 storage 76, 112, 172, 256; half-precision storage 48, 66, 96, 138.
#[test]
fn u23_memory_estimate_static_bytes_match_table() {
    let float32 = GpuLayoutOptions::default();
    let half = GpuLayoutOptions {
        half: ALL_HALF,
        sh_cap: None,
    };
    for (l, f32_bytes, half_bytes) in [
        (0u8, 76u64, 48u64),
        (1, 112, 66),
        (2, 172, 96),
        (3, 256, 138),
    ] {
        for n in [1u64, 1000, 3_000_001] {
            assert_eq!(
                memory_estimate(n, l, &float32, false).static_bytes,
                n * f32_bytes,
                "L = {l}"
            );
            assert_eq!(
                memory_estimate(n, l, &half, true).static_bytes,
                n * half_bytes,
                "L = {l}"
            );
        }
    }
    // §1.12's examples: 1.5 M primitives of degree 2 in `quest-high` take 144 MB static;
    // 3 M of degree 3 in `desktop` take 768 MB.
    assert_eq!(
        memory_estimate(1_500_000, 2, &half, true).static_bytes,
        144_000_000
    );
    assert_eq!(
        memory_estimate(3_000_000, 3, &float32, false).static_bytes,
        768_000_000
    );
}

/// §1.2 orders the SH cap (option 1) before the memory check (option 4): the estimate
/// counts the capped degree, and each half flag halves its own field.
#[test]
fn u23_memory_estimate_counts_capped_degree_and_each_flag() {
    let capped = GpuLayoutOptions {
        half: ALL_HALF,
        sh_cap: Some(2),
    };
    assert_eq!(memory_estimate(10, 3, &capped, true).static_bytes, 10 * 96);
    let above = GpuLayoutOptions {
        sh_cap: Some(3),
        ..Default::default()
    };
    assert_eq!(memory_estimate(10, 1, &above, false).static_bytes, 10 * 112);
    // float32 degree 0 is 76 bytes; velocity 12 → 6, scale 12 → 6, rotation 16 → 8,
    // opacity 4 → 2, sh 12 → 6.
    let only = |f: fn(&mut HalfFields)| {
        let mut half = HalfFields::default();
        f(&mut half);
        memory_estimate(1, 0, &GpuLayoutOptions { half, sh_cap: None }, false).static_bytes
    };
    assert_eq!(only(|h| h.velocity = true), 70);
    assert_eq!(only(|h| h.scale = true), 70);
    assert_eq!(only(|h| h.rotation = true), 68);
    assert_eq!(only(|h| h.opacity = true), 74);
    assert_eq!(only(|h| h.sh = true), 70);
}

/// U-23 and Q-T13.2: §1.12's frame buffers per unit of capacity C for one set (one view,
/// or the shared set): records 48 C per view or 64 C shared-center, sort keys and payloads
/// 16 C, histograms 1 C; and the layer's 8 bytes per pixel. §1.12's examples: C = 4 M per
/// view about 260 MB; C = 1.5 M shared about 122 MB; two 1680 × 1760 eye layers about
/// 47 MB.
#[test]
fn u23_memory_estimate_frame_buffers_and_layer() {
    for opts in [
        GpuLayoutOptions::default(),
        GpuLayoutOptions {
            half: ALL_HALF,
            sh_cap: Some(1),
        },
    ] {
        let per_view = memory_estimate(1, 3, &opts, false);
        let shared = memory_estimate(1, 3, &opts, true);
        assert_eq!(per_view.per_capacity_bytes, 48 + 16 + 1);
        assert_eq!(shared.per_capacity_bytes, 64 + 16 + 1);
        assert_eq!(per_view.layer_bytes_per_pixel, 8);
        assert_eq!(shared.layer_bytes_per_pixel, 8);
    }
    let MemoryEstimate {
        per_capacity_bytes,
        layer_bytes_per_pixel,
        ..
    } = memory_estimate(1, 3, &GpuLayoutOptions::default(), false);
    assert_eq!(u64::from(per_capacity_bytes) * 4_000_000, 260_000_000);
    let shared = memory_estimate(
        1,
        2,
        &GpuLayoutOptions {
            half: ALL_HALF,
            sh_cap: None,
        },
        true,
    );
    assert_eq!(
        u64::from(shared.per_capacity_bytes) * 1_500_000,
        121_500_000
    );
    assert_eq!(
        u64::from(layer_bytes_per_pixel) * 1680 * 1760 * 2,
        47_308_800
    );
}

/// The static estimate counts exactly what `pack_for_gpu` writes (Q-T13.1, Q-T13.2), for
/// an odd N and mixed flags too.
#[test]
fn u23_static_bytes_equal_packed_bytes() {
    let m = model(7, 0.0, 8.0);
    let order: Vec<u32> = (0..7).collect();
    let mixed = HalfFields {
        velocity: true,
        scale: false,
        rotation: true,
        opacity: true,
        sh: false,
    };
    for half in [HalfFields::default(), ALL_HALF, mixed] {
        for sh_cap in [None, Some(0)] {
            let opts = GpuLayoutOptions { half, sh_cap };
            let a = pack_for_gpu(&m, &order, &opts);
            let packed = 4 * (a.position.len() + a.time_center.len() + a.duration.len())
                + a.velocity.len()
                + a.scale.len()
                + a.rotation.len()
                + a.opacity.len()
                + a.sh.len();
            let estimate = memory_estimate(7, m.sh_degree, &opts, false).static_bytes;
            assert_eq!(estimate, packed as u64, "{half:?} cap {sh_cap:?}");
        }
    }
}

/// U-23: `reduce_to_budget` keeps the `budget` rows of highest opacity·scale_x·scale_y·
/// scale_z (§1.2 load option 4, D-13), returned in file order. Hand-worked scores: row 0
/// 0.5, row 1 0.09, row 2 0.8, row 3 0.5, row 4 0.1, row 5 0.7; rows 0 and 3 tie, and
/// the earlier file row is kept (Q-T13.3).
#[test]
fn u23_reduce_to_budget_keeps_top_scores_in_file_order() {
    let mut m = model(6, 0.0, 8.0);
    m.opacity = vec![0.5, 0.9, 0.1, 0.5, 0.2, 0.7];
    m.scale = vec![
        [1.0, 1.0, 1.0],
        [0.1, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 1.0],
    ];
    m.validate().unwrap();
    assert_eq!(reduce_to_budget(&m, 0), Vec::<u32>::new());
    assert_eq!(reduce_to_budget(&m, 1), [2]);
    assert_eq!(reduce_to_budget(&m, 2), [2, 5]);
    assert_eq!(reduce_to_budget(&m, 3), [0, 2, 5]);
    assert_eq!(reduce_to_budget(&m, 4), [0, 2, 3, 5]);
    assert_eq!(reduce_to_budget(&m, 5), [0, 2, 3, 4, 5]);
    assert_eq!(reduce_to_budget(&m, 6), [0, 1, 2, 3, 4, 5]);
    assert_eq!(reduce_to_budget(&m, 100), [0, 1, 2, 3, 4, 5]);
}

/// U-23: on 5,000 random rows with many ties, every kept score is at least every dropped
/// one, the rows are strictly ascending, and the result is the same on every call.
#[test]
fn u23_reduce_to_budget_top_scores_deterministic() {
    let n = 5000;
    let mut g = Values(0x0123_4567_89ab_cdef);
    let mut m = model(n, 0.0, 8.0);
    // Scores drawn from a few levels, so ties are common.
    m.opacity = (0..n)
        .map(|_| [0.25f32, 0.5, 0.75][(g.next(0.0, 3.0) as usize).min(2)])
        .collect();
    m.scale = (0..n)
        .map(|_| [g.next(0.5, 1.0).round().max(0.5), 1.0, 2.0])
        .collect();
    let score = |i: usize| {
        let s = m.scale[i];
        f64::from(m.opacity[i]) * f64::from(s[0]) * f64::from(s[1]) * f64::from(s[2])
    };
    for budget in [1, 17, 1000, 2500, 4999] {
        let kept = reduce_to_budget(&m, budget);
        assert_eq!(kept.len(), budget);
        assert!(kept.windows(2).all(|w| w[0] < w[1]), "file order");
        let mut is_kept = vec![false; n];
        kept.iter().for_each(|&i| is_kept[i as usize] = true);
        let lowest_kept = kept
            .iter()
            .map(|&i| score(i as usize))
            .fold(f64::INFINITY, f64::min);
        let highest_dropped = (0..n)
            .filter(|&i| !is_kept[i])
            .map(score)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(lowest_kept >= highest_dropped, "budget {budget}");
        assert_eq!(reduce_to_budget(&m, budget), kept, "deterministic");
    }
}
