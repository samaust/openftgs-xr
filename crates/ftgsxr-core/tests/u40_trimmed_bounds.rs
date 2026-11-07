// SPDX-License-Identifier: Apache-2.0

//! U-40 (§3.1, `cpu` lane): E17's trimmed boxes (`trimmed_bounds`) on hand-built models
//! packed in float32, against hand-worked literals (D-56). α_skip is openftgs's 1/255; a
//! row of opacity α_skip·e^(k²/2) has a visible half-width h = k·duration (E10).

use ftgsxr_core::{
    Aabb, GpuLayoutOptions, Model, ModelKind, RenderConventions, TimeSpan, Trim, TrimMode, Vec3,
    pack_for_gpu, trimmed_bounds,
};

/// U-40's tolerance, 10⁻⁶ absolute (the literals stay below 6 in magnitude, where float32
/// holds it, as U-38's).
fn assert_box(b: Aabb, min: [f32; 3], max: [f32; 3]) {
    let (min, max) = (Vec3::from_array(min), Vec3::from_array(max));
    let tol = 1e-6;
    assert!(
        (b.min - min).abs().max_element() <= tol && (b.max - max).abs().max_element() <= tol,
        "{b:?} vs {min} – {max}"
    );
}

const ALPHA_SKIP: f32 = 1.0 / 255.0;

/// The opacity whose visible half-width is `k` durations: α_skip·e^(k²/2).
fn opacity_for(k: f64) -> f32 {
    (f64::from(ALPHA_SKIP) * (k * k / 2.0).exp()) as f32
}

#[derive(Clone, Copy)]
struct Row {
    position: [f32; 3],
    time_center: f64,
    duration: f64,
    velocity: [f32; 3],
    scale: f32,
    opacity: f32,
}

/// A still row at `position`, visible over the whole span [0, 10].
fn still(position: [f32; 3], scale: f32) -> Row {
    Row {
        position,
        time_center: 5.0,
        duration: 1000.0,
        velocity: [0.0; 3],
        scale,
        opacity: 0.5,
    }
}

/// A row at the origin at τ = 5 moving at 10 units per unit of τ along x, duration 0.1
/// and h = 0.2: visible over [4.8, 5.2], where it runs from x = −2 to x = 2.
fn fast_short() -> Row {
    Row {
        position: [0.0; 3],
        time_center: 5.0,
        duration: 0.1,
        velocity: [10.0, 0.0, 0.0],
        scale: 0.01,
        opacity: opacity_for(2.0),
    }
}

fn model(rows: &[Row]) -> Model {
    let n = rows.len();
    Model {
        position: rows.iter().map(|r| r.position).collect(),
        time_center: rows.iter().map(|r| r.time_center).collect(),
        duration: rows.iter().map(|r| r.duration).collect(),
        velocity: rows.iter().map(|r| r.velocity).collect(),
        scale: rows.iter().map(|r| [r.scale; 3]).collect(),
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacity: rows.iter().map(|r| r.opacity).collect(),
        sh: vec![0.0; n * 3],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 10.0,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// `trimmed_bounds` of the rows packed in float32, in file order.
fn trimmed(rows: &[Row], trim: &Trim) -> Option<(Aabb, Aabb)> {
    let m = model(rows);
    let order: Vec<u32> = (0..m.len() as u32).collect();
    let arrays = pack_for_gpu(&m, &order, &GpuLayoutOptions::default());
    trimmed_bounds(&arrays, ALPHA_SKIP, trim)
}

fn boxed(min: [f32; 3], max: [f32; 3]) -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: Vec3::from_array(min),
        box_max: Vec3::from_array(max),
        center: Vec3::ZERO,
        radius: 1.0,
    }
}

fn sphere(center: [f32; 3], radius: f32) -> Trim {
    Trim {
        mode: TrimMode::Sphere,
        box_min: Vec3::ZERO,
        box_max: Vec3::ZERO,
        center: Vec3::from_array(center),
        radius,
    }
}

#[test]
fn u40_still_rows_inside_and_outside_a_box_and_a_sphere() {
    let rows = [still([0.0; 3], 0.01), still([5.0, 0.0, 0.0], 0.01)];
    // The box keeps the first; one kept row: B_c is the trimmed B_m.
    let (m, c) = trimmed(&rows, &boxed([-1.0; 3], [1.0; 3])).unwrap();
    assert_box(m, [-0.03; 3], [0.03; 3]);
    assert_eq!(c, m);
    // The sphere keeps the second.
    let (m, c) = trimmed(&rows, &sphere([5.0, 0.0, 0.0], 1.0)).unwrap();
    assert_box(m, [4.97, -0.03, -0.03], [5.03, 0.03, 0.03]);
    assert_eq!(c, m);
}

#[test]
fn u40_a_moving_row_partly_inside_covers_its_kept_interval_only() {
    // Box x ∈ [−1, 3]: of x ∈ [−2, 2] it keeps [−1, 2] (τ ∈ [4.9, 5.2]).
    let (m, c) = trimmed(&[fast_short()], &boxed([-1.0, -1.0, -1.0], [3.0, 1.0, 1.0])).unwrap();
    assert_box(m, [-1.03, -0.03, -0.03], [2.03, 0.03, 0.03]);
    assert_eq!(c, m);
    // Sphere of radius 1 at x = 1.5: keeps x ∈ [0.5, 2] (the visible interval's end).
    let (m, c) = trimmed(&[fast_short()], &sphere([1.5, 0.0, 0.0], 1.0)).unwrap();
    assert_box(m, [0.47, -0.03, -0.03], [2.03, 0.03, 0.03]);
    assert_eq!(c, m, "one kept row: B_c is the trimmed B_m");
    // With a still row inside the sphere too, B_c is the box of the moving row's kept
    // midpoint (x = 1.25 at τ = 5.125) and the still row (x = 2.2).
    let rows = [fast_short(), still([2.2, 0.0, 0.0], 0.01)];
    let (_, c) = trimmed(&rows, &sphere([1.5, 0.0, 0.0], 1.0)).unwrap();
    assert_box(c, [1.25, 0.0, 0.0], [2.2, 0.0, 0.0]);
    // The midpoints of the kept intervals make B_c: with a second row the box of the two
    // midpoints (x = 0.5 at τ = 5.05, and a still row at x = −0.9).
    let rows = [fast_short(), still([-0.9, 0.0, 0.0], 0.01)];
    let (m, c) = trimmed(&rows, &boxed([-1.0, -1.0, -1.0], [3.0, 1.0, 1.0])).unwrap();
    assert_box(m, [-1.03, -0.03, -0.03], [2.03, 0.03, 0.03]);
    assert_box(c, [-0.9, 0.0, 0.0], [0.5, 0.0, 0.0]);
}

#[test]
fn u40_a_row_passing_through_with_both_ends_outside_is_kept() {
    // x ∈ [−2, 2] through x ∈ [−0.5, 0.5]: both visible ends lie outside.
    let (m, _) = trimmed(&[fast_short()], &boxed([-0.5, -1.0, -1.0], [0.5, 1.0, 1.0])).unwrap();
    assert_box(m, [-0.53, -0.03, -0.03], [0.53, 0.03, 0.03]);
    // Likewise through a sphere of radius 0.5 at the origin.
    let (m, _) = trimmed(&[fast_short()], &sphere([0.0; 3], 0.5)).unwrap();
    assert_box(m, [-0.53, -0.03, -0.03], [0.53, 0.03, 0.03]);
}

#[test]
fn u40_a_row_at_or_below_alpha_skip_inside_is_ignored() {
    let faint = Row {
        opacity: 0.001, // < 1/255
        ..still([0.5, 0.5, 0.5], 0.01)
    };
    let rows = [still([0.0; 3], 0.01), faint, still([9.0, 0.0, 0.0], 0.01)];
    let (m, _) = trimmed(&rows, &boxed([-1.0; 3], [1.0; 3])).unwrap();
    assert_box(m, [-0.03; 3], [0.03; 3]);
    // Alone inside, with the counted row outside: nothing is kept.
    assert_eq!(
        trimmed(
            &[faint, still([9.0, 0.0, 0.0], 0.01)],
            &boxed([-1.0; 3], [1.0; 3])
        ),
        None
    );
}

#[test]
fn u40_the_expansion_is_taken_over_the_kept_rows_only() {
    // A huge primitive outside the trim widens nothing; a large one inside widens only
    // its own extent.
    let rows = [
        still([0.0; 3], 0.01),
        still([50.0, 0.0, 0.0], 10.0),
        still([0.5, 0.0, 0.0], 0.1),
    ];
    let (m, _) = trimmed(&rows, &boxed([-1.0; 3], [1.0; 3])).unwrap();
    assert_box(m, [-0.03, -0.3, -0.3], [0.8, 0.3, 0.3]);
}

#[test]
fn u40_the_content_box_takes_the_percentiles_of_the_kept_rows() {
    // 100 kept rows at x = 0.05 k (k = 1 … 100) and 100 trimmed ones at x = 5.5 + 0.004 k:
    // over the 100 kept, the nearest ranks ⌈1⌉ and ⌈99⌉ are x = 0.05 and x = 4.95.
    let mut rows: Vec<Row> = (1..=100)
        .map(|k| still([0.05 * k as f32, 0.0, 0.0], 0.01))
        .collect();
    rows.extend((1..=100).map(|k| still([5.5 + 0.004 * k as f32, 0.0, 0.0], 0.01)));
    let (m, c) = trimmed(&rows, &boxed([0.025, -1.0, -1.0], [5.025, 1.0, 1.0])).unwrap();
    assert_box(m, [0.02, -0.03, -0.03], [5.03, 0.03, 0.03]);
    assert_box(c, [0.05, 0.0, 0.0], [4.95, 0.0, 0.0]);
}

#[test]
fn u40_no_row_kept_or_the_trim_off_gives_none() {
    let rows = [still([0.0; 3], 0.01), fast_short()];
    assert_eq!(trimmed(&rows, &boxed([5.0; 3], [6.0; 3])), None);
    assert_eq!(trimmed(&rows, &sphere([0.0, 3.0, 0.0], 1.0)), None);
    // Off keeps everything, but `trimmed_bounds` answers only for a trim that is on.
    let off = Trim {
        mode: TrimMode::Off,
        ..boxed([-10.0; 3], [10.0; 3])
    };
    assert_eq!(trimmed(&rows, &off), None);
}

#[test]
fn u40_with_no_counted_row_every_row_counts_over_the_span() {
    // Both rows are below α_skip, so E10's fallback counts both over [0, 10]; the box
    // keeps the first. The moving row at 0.1 units per unit of τ from x = 2 − 0.5 = 1.5 at
    // τ = 0 to x = 2.5 at τ = 10: kept over x ∈ [1.5, 2].
    let a = Row {
        opacity: 0.001,
        ..still([0.0; 3], 0.01)
    };
    let b = Row {
        position: [2.0, 0.0, 0.0],
        velocity: [0.1, 0.0, 0.0],
        opacity: 0.001,
        ..still([0.0; 3], 0.01)
    };
    let (m, _) = trimmed(&[a, b], &boxed([-1.0, -1.0, -1.0], [2.0, 1.0, 1.0])).unwrap();
    assert_box(m, [-0.03, -0.03, -0.03], [2.03, 0.03, 0.03]);
}
