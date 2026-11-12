// SPDX-License-Identifier: Apache-2.0

//! B-4 (§3.5): the cull's GPU time and the rows it visits, with the temporal index (E3)
//! and with the cull dispatched over every row, on both machines (Q-408, Q-430). The
//! every-row run is the bench's own: the same packed rows uploaded with the index of a
//! single-timestamp model, whose dispatch range is every row at every τ, so the cull
//! still decides activity by E3's test (D-10) and `ftgsxr_render` needs no switch.

use ftgsxr_core::{
    DispatchRange, Model, ModelKind, RenderConventions, TemporalIndex, TimeSpan, pack_for_gpu,
};
use ftgsxr_render::{FrameBuffers, FrameStats, GpuModel};

use super::{
    Budget, Given, Gpu, HeadlessError, Json, Measured, On, Rule, Targets, WARM_UP, draw, frame,
    gpu_error, median, orbit, records_refusal, refused, with_n,
};
use crate::benchmark::synthetic::{Persistence, synthetic_model};

/// B-4's sizes N (§3.5: 10⁵ to 3 × 10⁶; Q-430).
// Q-430
pub(super) const B4_N: [usize; 6] = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000];
/// The share of the every-row median the index may be slower by when the run's spread is
/// smaller (Q-430).
// Q-430
const B4_RELATIVE_TOLERANCE: f64 = 0.1;
/// B-4's measured frames per setup, as B-3's (Q-102, Q-430).
const B4_FRAMES: usize = 32;
/// The `model` label of B-4's synthetic models: `random_model`'s rows, mostly transient
/// (Q-408).
const SYNTHETIC_MIXED: &str = "synthetic, random_model's rows";
/// The two setups of a size: the temporal index, then every row (Q-430).
const INDEXED: &str = "temporal";
/// As `INDEXED`.
const EVERY_ROW: &str = "every row";
/// The note on B-4's figures.
const B4_NOTE: &str = "B-4's cull_ms is the median GPU time of stages 1-3 (the cull and the one-workgroup size dispatch) over 32 frames, cull_ms_spread their largest minus their smallest; rows_visited is the dispatch range at the frame's tau (the temporal index's, or every row); the B-4 budgets check that the indexed cull is not slower than the every-row cull by more than the tolerance, the larger of the two setups' spreads and 10 % of the every-row median, and visits no more rows (Q-408, Q-430)";

/// The temporal index of `n` rows that visits every row at every τ: a single-timestamp
/// model's, every row persistent (E3, `TemporalIndex::build`). Only its range is used:
/// the rows uploaded with it keep the model's own order.
// Q-430
pub(super) fn every_row_index(n: usize) -> TemporalIndex {
    let model = Model {
        position: vec![[0.0; 3]; n],
        time_center: vec![0.0; n],
        duration: vec![1.0; n],
        velocity: vec![[0.0; 3]; n],
        scale: vec![[1.0; 3]; n],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacity: vec![0.5; n],
        sh: vec![0.0; 3 * n],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 0.0,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    };
    TemporalIndex::build(&model).0
}

/// The rows a cull at `tau` visits through `index`: its dispatch range's (E3), the count
/// the cull pass dispatches.
pub(super) fn rows_visited(index: &TemporalIndex, tau: f64) -> u32 {
    let DispatchRange {
        persistent,
        transient,
    } = index.range(tau);
    persistent.len() as u32 + transient.len() as u32
}

/// One setup's figures (Q-430).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Cull {
    /// The median cull time, ms.
    pub(super) median_ms: f64,
    /// The largest minus the smallest cull time, ms.
    pub(super) spread_ms: f64,
    /// The rows the cull visits.
    pub(super) rows: u32,
}

impl Cull {
    /// The figures of the cull times `ms` and the rows visited.
    pub(super) fn of(ms: &mut [f64], rows: u32) -> Cull {
        let lo = ms.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Cull {
            median_ms: median(ms),
            spread_ms: if ms.is_empty() { 0.0 } else { hi - lo },
            rows,
        }
    }
}

/// B-4's two checks at one size of N rows, on the `temporal` setup (§3.5 "the index never
/// slower", Q-408): the indexed cull time at most the every-row one plus the tolerance,
/// the larger of the run's spread (the larger of the two setups' spreads) and 10 % of the
/// every-row median (the owner's option 2 on #733); the rows visited at most N.
// Q-430
pub(super) fn b4_budgets(setup: &Json, n: usize, indexed: &Cull, every: &Cull) -> [Budget; 2] {
    let tolerance = indexed
        .spread_ms
        .max(every.spread_ms)
        .max(B4_RELATIVE_TOLERANCE * every.median_ms);
    [
        Budget {
            id: "B-4",
            setup: setup.clone(),
            measure: "cull_ms",
            rule: Rule::AtMost,
            limit: every.median_ms + tolerance,
            value: indexed.median_ms,
        },
        Budget {
            id: "B-4",
            setup: setup.clone(),
            measure: "rows_visited",
            rule: Rule::AtMost,
            limit: n as f64,
            value: f64::from(indexed.rows),
        },
    ]
}

/// B-4's setup (Q-430).
fn b4_setup(label: Json, n: usize, tier: &str, index: &str) -> Json {
    Json::obj([
        ("model", label),
        ("n", Json::Num(n as f64)),
        ("sh_degree", Json::Num(0.0)),
        ("tier", Json::str(tier)),
        ("index", Json::str(index)),
    ])
}

/// The measures of a B-4 setup, with their units.
const MEASURES: [(&str, &str); 5] = [
    ("cull_ms", "ms"),
    ("cull_ms_spread", "ms"),
    ("rows_visited", "rows"),
    ("visible_splats", "splats"),
    ("overflow", "flag"),
];

/// B-4 (§3.5, Q-408, Q-430): the cull with and without the temporal index; reported,
/// with the check that the index is never slower.
// Q-408, Q-430
pub(super) fn bench_b4(gpu: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let mut m = Measured {
        notes: vec![B4_NOTE],
        ..Measured::default()
    };
    let cases: Vec<usize> = match given {
        Some(g) => vec![g.model.len()],
        None => B4_N.to_vec(),
    };
    let targets = Targets::new(gpu);
    let tier = gpu.tier.name.clone();
    for n in cases {
        let synthetic;
        let (model, label) = match given {
            Some(g) => (&g.model, g.label()),
            None => {
                synthetic = synthetic_model(1, n, 0, Persistence::Few);
                (&synthetic, Json::str(SYNTHETIC_MIXED))
            }
        };
        let setups = [INDEXED, EVERY_ROW].map(|i| b4_setup(label.clone(), n, &tier, i));
        let skip_all = |m: &mut Measured, reason: String| {
            for setup in &setups {
                for (measure, unit) in MEASURES {
                    m.skip(setup, measure, unit, reason.clone());
                }
            }
        };
        let capacity = gpu.capacity(u32::try_from(n).unwrap_or(u32::MAX));
        // Q-413: a size the headset refuses is a result without a value.
        if matches!(gpu.on, On::Headset(_)) {
            let limits = gpu.dev.device().limits();
            if let Some(reason) = records_refusal(
                capacity,
                &gpu.tier,
                limits.max_storage_buffer_binding_size,
                limits.max_buffer_size,
            ) {
                skip_all(&mut m, reason);
                continue;
            }
        }
        let (index, order) = TemporalIndex::build(model);
        let arrays = pack_for_gpu(model, &order, &gpu.tier.layout);
        let loaded = arrays.sh_degree;
        let frames = FrameBuffers::new(&gpu.dev, &gpu.tier, capacity);
        let camera = orbit(model)[0].0;
        let tau = (model.time.tau0 + model.time.tau1) / 2.0;
        let file = given.map(|g| g.path.as_path());
        let mut culls = Vec::new();
        let mut refusal = None;
        for (setup, index) in setups.iter().zip([index, every_row_index(n)]) {
            let rows = rows_visited(&index, tau);
            // The rows stay in the model's order for both (§2.4 GPU data layout).
            let gpu_model = match GpuModel::upload(&gpu.dev, &arrays, index, &model.conventions)
                .map_err(|e| with_n(gpu_error(file, e), n))
            {
                Ok(uploaded) => uploaded,
                Err(e) if refused(&gpu.on, &e) => {
                    refusal = Some(e.to_string());
                    break;
                }
                Err(e) => return Err(e),
            };
            let stats = measured_frames(gpu, &frames, &gpu_model, loaded, &camera, tau, &targets)?;
            let mut ms: Vec<f64> = stats
                .iter()
                .map(|s| f64::from(s.gpu_times_ms.cull))
                .collect();
            let mut visible: Vec<f64> = stats.iter().map(|s| f64::from(s.visible_splats)).collect();
            let overflow = stats.iter().any(|s| s.overflow);
            let cull = Cull::of(&mut ms, rows);
            m.figure(setup, "cull_ms", cull.median_ms, "ms");
            m.figure(setup, "cull_ms_spread", cull.spread_ms, "ms");
            m.figure(setup, "rows_visited", f64::from(rows), "rows");
            m.figure(setup, "visible_splats", median(&mut visible), "splats");
            m.figure(setup, "overflow", f64::from(u8::from(overflow)), "flag");
            culls.push(cull);
        }
        match (refusal, culls.as_slice()) {
            // Q-413: a refused upload skips the size in both setups.
            (Some(reason), _) => {
                m.figures.retain(|f| !setups.contains(&f.setup));
                skip_all(&mut m, reason);
            }
            (None, [indexed, every]) => {
                m.budgets.extend(b4_budgets(&setups[0], n, indexed, every));
            }
            (None, _) => unreachable!("both setups ran"),
        }
    }
    Ok(m)
}

/// B-3's frame loop for one setup (Q-102): 8 warm-up frames, then the statistics of the
/// 32 measured frames (read two frames late, so two more frames are run); no timestamp
/// queries is E-GPU-ADAPTER.
fn measured_frames(
    gpu: &Gpu,
    frames: &FrameBuffers,
    model: &GpuModel,
    loaded: u8,
    camera: &ftgsxr_core::Camera,
    tau: f64,
    targets: &Targets,
) -> Result<Vec<FrameStats>, HeadlessError> {
    let mut stats = Vec::new();
    for i in 0..WARM_UP + B4_FRAMES + 2 {
        let (_, s) = frame(gpu, frames, &[draw(model, loaded, tau)], camera, targets)?;
        if i >= WARM_UP {
            stats.extend(s);
        }
    }
    let stats = stats.split_off(stats.len().saturating_sub(B4_FRAMES));
    if stats.is_empty() || stats.iter().all(|s| s.gpu_times_ms.cull == 0.0) {
        return Err(HeadlessError::Environment(
            "E-GPU-ADAPTER: the adapter has no timestamp queries, so B-4 cannot time the cull"
                .to_owned(),
        ));
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b4_sizes_run_from_1e5_to_3e6() {
        // §3.5 B-4: rows visited from 10⁵ to 3 × 10⁶.
        assert_eq!(B4_N.first(), Some(&100_000));
        assert_eq!(B4_N.last(), Some(&3_000_000));
        assert!(B4_N.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn b4_every_row_index_visits_every_row_at_every_tau() {
        let index = every_row_index(1000);
        assert_eq!(index.len(), 1000);
        for tau in [-5.0, 0.0, 0.95, 1.9, 7.0] {
            let range = index.range(tau);
            assert_eq!(range.persistent, 0..1000, "{tau}");
            assert!(range.transient.is_empty(), "{tau}");
            assert_eq!(rows_visited(&index, tau), 1000);
        }
    }

    #[test]
    fn b4_the_mixed_model_visits_fewer_rows_with_the_index_at_mid_span() {
        // Q-408: random_model's transient rows are what the index exists for.
        let model = synthetic_model(1, 20_000, 0, Persistence::Few);
        let (index, _) = TemporalIndex::build(&model);
        let tau = (model.time.tau0 + model.time.tau1) / 2.0;
        let rows = rows_visited(&index, tau);
        assert!(rows < 20_000, "{rows}");
        assert!(rows >= 4, "{rows}");
    }

    #[test]
    fn b4_cull_takes_the_median_and_the_full_range() {
        let cull = Cull::of(&mut [0.3, 0.1, 0.2, 0.6, 0.2], 7);
        assert_eq!(cull.median_ms, 0.2);
        assert!((cull.spread_ms - 0.5).abs() < 1e-12);
        assert_eq!(cull.rows, 7);
    }

    #[test]
    fn b4_budgets_allow_the_index_the_larger_spread_and_no_more_rows() {
        let setup = Json::obj([("n", Json::Num(100.0))]);
        let every = Cull {
            median_ms: 1.0,
            spread_ms: 0.1,
            rows: 100,
        };
        // Slower by 0.2 ms, within its own spread of 0.25: passes.
        let indexed = Cull {
            median_ms: 1.2,
            spread_ms: 0.25,
            rows: 40,
        };
        let [time, rows] = b4_budgets(&setup, 100, &indexed, &every);
        assert_eq!(
            (time.id, time.measure, time.rule),
            ("B-4", "cull_ms", Rule::AtMost)
        );
        assert!((time.limit - 1.25).abs() < 1e-12);
        assert_eq!(time.value, 1.2);
        assert!(time.pass());
        assert_eq!(
            (rows.measure, rows.limit, rows.value),
            ("rows_visited", 100.0, 40.0)
        );
        assert!(rows.pass());
        // Slower by more than both spreads: fails.
        let slow = Cull {
            median_ms: 1.3,
            ..indexed.clone()
        };
        assert!(!b4_budgets(&setup, 100, &slow, &every)[0].pass());
        // More rows than N: fails.
        let more = Cull {
            rows: 101,
            ..indexed
        };
        assert!(!b4_budgets(&setup, 100, &more, &every)[1].pass());
    }

    #[test]
    fn b4_budgets_allow_10_percent_of_the_every_row_median_when_the_spreads_are_smaller() {
        // Q-430 (owner, option 2): the tolerance is the larger of the run's spread and
        // 10 % of the every-row median: here 0.008 ms against 0.01 ms, so the limit is
        // 0.1 + 0.01 = 0.11 ms.
        let setup = Json::obj([("n", Json::Num(100.0))]);
        let every = Cull {
            median_ms: 0.1,
            spread_ms: 0.005,
            rows: 100,
        };
        let indexed = Cull {
            median_ms: 0.108,
            spread_ms: 0.008,
            rows: 40,
        };
        let [time, _] = b4_budgets(&setup, 100, &indexed, &every);
        assert!((time.limit - 0.11).abs() < 1e-12, "{}", time.limit);
        assert!(time.pass());
        let slow = Cull {
            median_ms: 0.111,
            ..indexed
        };
        assert!(!b4_budgets(&setup, 100, &slow, &every)[0].pass());
    }
}
