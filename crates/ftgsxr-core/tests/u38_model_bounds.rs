// SPDX-License-Identifier: Apache-2.0

//! U-38 (§3.1, `cpu` lane): E10's B_m (`Model::bounds`) and content box B_c
//! (`Model::content_bounds`) on hand-built models, against hand-worked literals (D-42).
//! α_skip is openftgs's 1/255; a row of opacity α_skip·e^(k²/2) has a visible half-width
//! h = k·duration.

use ftgsxr_core::{Aabb, Model, ModelKind, RenderConventions, TimeSpan, Vec3};

/// U-38's tolerance, 10⁻⁶ (the literals stay below 10 in magnitude, where float32 holds
/// it).
fn assert_vec3(a: Vec3, b: Vec3) {
    assert!((a - b).abs().max_element() <= 1e-6, "{a} vs {b}");
}

fn assert_box(b: Aabb, min: [f32; 3], max: [f32; 3]) {
    assert_vec3(b.min, Vec3::from_array(min));
    assert_vec3(b.max, Vec3::from_array(max));
}

/// The opacity whose visible half-width is `k` durations: α_skip·e^(k²/2).
fn opacity_for(k: f64) -> f32 {
    (f64::from(1.0f32 / 255.0) * (k * k / 2.0).exp()) as f32
}

/// One row of a hand-built model.
#[derive(Clone, Copy)]
struct Row {
    position: [f32; 3],
    time_center: f64,
    duration: f64,
    velocity: [f32; 3],
    scale: f32,
    opacity: f32,
}

/// A static row at `position`, visible over any span of a few hundred units.
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

fn model(rows: &[Row], tau0: f64, tau1: f64) -> Model {
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
            tau0,
            tau1,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// A row at the origin at τ = 5 moving at 10 units per unit of τ along x, duration 0.1
/// and h = 0.2: visible over [4.8, 5.2], where it lies at x = ∓2.
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

#[test]
fn u38_a_fast_short_row_adds_its_visible_interval_ends_only() {
    // Over [0, 10] its extrapolated positions would reach x = ±50.
    let b = model(&[fast_short()], 0.0, 10.0).bounds();
    assert_box(b, [-2.03, -0.03, -0.03], [2.03, 0.03, 0.03]);
}

#[test]
fn u38_rows_below_alpha_skip_or_outside_the_span_add_nothing() {
    let faint = Row {
        opacity: 0.001, // < 1/255
        ..still([100.0, 0.0, 0.0], 0.01)
    };
    // h = 2 about τ = 20: [18, 22] misses [0, 10].
    let late = Row {
        position: [-100.0, 0.0, 0.0],
        time_center: 20.0,
        duration: 1.0,
        velocity: [0.0; 3],
        scale: 0.01,
        opacity: opacity_for(2.0),
    };
    let b = model(&[fast_short(), faint, late], 0.0, 10.0).bounds();
    assert_box(b, [-2.03, -0.03, -0.03], [2.03, 0.03, 0.03]);
}

#[test]
fn u38_a_visible_interval_past_tau1_is_clipped_there() {
    // h = 2 about τ = 9.5: [7.5, 11.5] clipped to [7.5, 10]; y = τ − 9.5 goes from −2 to
    // 0.5, not to 2.
    let row = Row {
        position: [0.0; 3],
        time_center: 9.5,
        duration: 1.0,
        velocity: [0.0, 1.0, 0.0],
        scale: 0.01,
        opacity: opacity_for(2.0),
    };
    let b = model(&[row], 0.0, 10.0).bounds();
    assert_box(b, [-0.03, -2.03, -0.03], [0.03, 0.53, 0.03]);
}

#[test]
fn u38_a_large_primitive_widens_only_its_own_extent() {
    // The large row at x = 10 with scale 3 spans x ∈ [1, 19], y and z ∈ [−9, 9]; the
    // small one at the origin keeps its 0.03 margin.
    let b = model(
        &[still([0.0; 3], 0.01), still([10.0, 0.0, 0.0], 3.0)],
        0.0,
        10.0,
    )
    .bounds();
    assert_box(b, [-0.03, -9.0, -9.0], [19.0, 9.0, 9.0]);
}

#[test]
fn u38_a_single_timestamp_model_counts_every_visible_row_at_tau0() {
    let rows = [
        still([0.0; 3], 0.1),
        still([1.0, 2.0, 3.0], 0.1),
        Row {
            opacity: 0.001,
            ..still([50.0, 0.0, 0.0], 0.1)
        },
    ]
    .map(|r| Row {
        time_center: 0.0,
        duration: 1.0,
        ..r
    });
    let b = model(&rows, 0.0, 0.0).bounds();
    assert_box(b, [-0.3, -0.3, -0.3], [1.3, 2.3, 3.3]);
}

#[test]
fn u38_with_no_counted_row_every_row_counts_over_the_span() {
    // D-42's fallback: both rows are below α_skip; the moving one spans x ∈ [−5, 5] over
    // [0, 10], each with its own margin.
    let moving = Row {
        position: [0.0; 3],
        time_center: 5.0,
        duration: 0.1,
        velocity: [1.0, 0.0, 0.0],
        scale: 0.1,
        opacity: 0.001,
    };
    let big = Row {
        opacity: 0.002,
        ..still([0.0, 4.0, 0.0], 1.0)
    };
    let m = model(&[moving, big], 0.0, 10.0);
    assert_box(m.bounds(), [-5.3, -0.3, -3.0], [5.3, 7.0, 3.0]);
    // B_c at the span's midpoint τ = 5: the two rows at (0, 0, 0) and (0, 4, 0).
    assert_box(m.content_bounds(), [0.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
}

#[test]
fn u38_content_faces_are_the_nearest_rank_percentiles_outliers_beyond() {
    // n = 200: the 2nd and 198th smallest. Rows at x = (k − 100)/20 for k = 2 … 199,
    // from −4.9 to 4.95, and two outliers at ±9.
    let mut rows: Vec<Row> = (2..200)
        .map(|k: i32| still([(k - 100) as f32 / 20.0, 0.0, 0.0], 0.01))
        .collect();
    rows.push(still([-9.0, 0.0, 0.0], 0.01));
    rows.push(still([9.0, 0.0, 0.0], 0.01));
    let m = model(&rows, 0.0, 10.0);
    assert_box(m.content_bounds(), [-4.9, 0.0, 0.0], [4.9, 0.0, 0.0]);
    assert_box(m.bounds(), [-9.03, -0.03, -0.03], [9.03, 0.03, 0.03]);

    // n = 150: ⌈1.5⌉ = 2 and ⌈148.5⌉ = 149, on each axis independently: rows at
    // (k/20, −k/20, k/40), k = 1 … 150.
    let rows: Vec<Row> = (1..=150)
        .map(|k| {
            let k = k as f32;
            still([k / 20.0, -k / 20.0, k / 40.0], 0.01)
        })
        .collect();
    assert_box(
        model(&rows, 0.0, 10.0).content_bounds(),
        [0.1, -7.45, 0.05],
        [7.45, -0.1, 3.725],
    );
}

#[test]
fn u38_content_takes_each_row_at_its_visible_intervals_midpoint() {
    // The clipped row of the clipping test: [7.5, 10], midpoint 8.75, where y = −0.75;
    // a still row at y = 1. With n = 2 the percentiles are the minimum and the maximum.
    let clipped = Row {
        position: [0.0; 3],
        time_center: 9.5,
        duration: 1.0,
        velocity: [0.0, 1.0, 0.0],
        scale: 0.01,
        opacity: opacity_for(2.0),
    };
    let m = model(&[clipped, still([0.0, 1.0, 0.0], 0.01)], 0.0, 10.0);
    assert_box(m.content_bounds(), [0.0, -0.75, 0.0], [0.0, 1.0, 0.0]);
}

#[test]
fn u38_content_is_bounds_for_a_single_row() {
    let m = model(&[fast_short()], 0.0, 10.0);
    assert_eq!(m.content_bounds(), m.bounds());
    // Every counted row at one point: the same.
    let m = model(&[still([1.0; 3], 0.2), still([1.0; 3], 0.5)], 0.0, 10.0);
    assert_eq!(m.content_bounds(), m.bounds());
    assert_box(m.bounds(), [-0.5; 3], [2.5; 3]);
}
