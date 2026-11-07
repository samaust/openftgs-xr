// SPDX-License-Identifier: Apache-2.0

//! The exchange model in memory and its reader rules (§1.2), its header-only information
//! (§1.5) and the box of E10.

use glam::{DVec3, Vec3};
use rayon::prelude::*;

use super::{CoreError, RenderConventions, TimeSpan};

/// An exchange model in memory (§1.2): physical values in raw time units; row i is
/// primitive i. Every reader produces one, validated (`Model::validate`).
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// \[N, 3\] μx, world units of the camera poses.
    pub position: Vec<[f32; 3]>,
    /// \[N\] τ₀ + μt·T_τ, raw τ.
    pub time_center: Vec<f64>,
    /// \[N\] > 0, units of τ (1 with a single timestamp, openftgs D-70).
    pub duration: Vec<f64>,
    /// \[N, 3\] world units per unit of τ.
    pub velocity: Vec<[f32; 3]>,
    /// \[N, 3\] > 0, world units.
    pub scale: Vec<[f32; 3]>,
    /// \[N, 4\] unit quaternion, file order (w, x, y, z), canonical sign (§1.2, §1.14).
    pub rotation: Vec<[f32; 4]>,
    /// \[N\] in (0, 1).
    pub opacity: Vec<f32>,
    /// N · (L+1)² · 3, coefficient-major, RGB innermost (E4's order).
    pub sh: Vec<f32>,
    /// L ∈ 0..=3.
    pub sh_degree: u8,
    /// τ₀, τ₁ and the unit of τ.
    pub time: TimeSpan,
    /// The `render_conventions` block (D-39).
    pub conventions: RenderConventions,
    /// The producing software: empty when an exchange or PLY file names none, `openftgs`
    /// for a checkpoint whose manifest names no openftgs version (Q-34).
    pub producer: String,
    /// The SHA-256 of the file's bytes (D-40).
    pub identity: [u8; 32],
    /// Which reader produced the model, with a checkpoint's hints.
    pub kind: ModelKind,
}

/// Which of §1.2's three files a model came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelKind {
    /// An exchange model `ftgs-if`.
    ExchangeFile,
    /// A checkpoint `ftgs-checkpoint`, with its placement and frame-step hints (§1.5, §1.6).
    Checkpoint {
        /// The checkpoint's `spatial_scale` S (openftgs D-27).
        spatial_scale: f64,
        /// The checkpoint's `frame_dt` Δt_f in normalized time (openftgs E14).
        frame_dt: f64,
    },
    /// A PLY snapshot (D-41).
    Ply,
}

/// What the file's header alone tells (§1.5 header-only information).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// N, the number of primitives.
    pub n: u64,
    /// L.
    pub sh_degree: u8,
    /// τ₀, τ₁ and the unit of τ.
    pub time: TimeSpan,
    /// The producing software, empty when the file names none.
    pub producer: String,
    /// The `render_conventions` block.
    pub conventions: RenderConventions,
    /// Which kind of file it is.
    pub kind: ModelKind,
    /// The file's size in bytes.
    pub file_bytes: u64,
}

/// An axis-aligned box (E10's B_m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// The lower corner.
    pub min: Vec3,
    /// The upper corner.
    pub max: Vec3,
}

impl Aabb {
    /// The box's center, (min + max)/2.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The box's extent along each axis, max − min.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The longest of the three sides (§1.5's fit).
    pub fn longest_side(&self) -> f32 {
        self.size().max_element()
    }
}

/// E1's moved position μx(τ) = position + velocity (τ − time_center), computed in float64
/// and rounded to float32. Private to the core; the `motion` module's public E1 may reuse it.
pub(crate) fn moved_position_f64(
    position: [f32; 3],
    velocity: [f32; 3],
    time_center: f64,
    tau: f64,
) -> Vec3 {
    let dt = tau - time_center;
    let p = DVec3::from_array(position.map(f64::from));
    let v = DVec3::from_array(velocity.map(f64::from));
    (p + v * dt).as_vec3()
}

/// Rows from which the model's per-row work runs in parallel with rayon: `validate`'s
/// checks here, the readers' conversions in `loader` (PRD #1, "Reader internals").
pub(crate) const PARALLEL_ROWS: usize = 1 << 16;

/// The lowest index of `values` whose element is `broken`, searched in parallel when
/// `parallel`; either way the same index.
fn first_broken<T: Sync>(
    values: &[T],
    parallel: bool,
    broken: impl Fn(&T) -> bool + Sync + Send,
) -> Option<usize> {
    if parallel {
        values.par_iter().position_first(broken)
    } else {
        values.iter().position(broken)
    }
}

/// The first non-finite value of a slice, as an E-FILE-VALUE error naming the key and its
/// index in the flat slice.
fn check_finite<T: Copy + Into<f64> + Sync>(
    key: &str,
    values: &[T],
    parallel: bool,
) -> Result<(), CoreError> {
    match first_broken(values, parallel, |&x| !x.into().is_finite()) {
        Some(i) => Err(CoreError::Value(format!(
            "{key}[{i}] = {}",
            values[i].into()
        ))),
        None => Ok(()),
    }
}

#[expect(
    clippy::len_without_is_empty,
    reason = "§2.1 gives `len` without `is_empty`; a validated model is never empty (N ≥ 1)"
)]
impl Model {
    /// N, the number of primitives.
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// E10's B_m in model space: the box over the counted rows, each adding its
    /// positions μx at its visible interval's two ends, expanded on every side by 3 · its
    /// own largest scale. A row with opacity σ > α_skip is counted over
    /// \[time_center − h, time_center + h\] ∩ \[τ₀, τ₁\], h = duration·√(2 ln(σ/α_skip))
    /// (E2's cull, without the temporal index's widening), and not counted when that
    /// interval is empty; with a single timestamp every row with σ > α_skip is counted at
    /// τ₀. When no row is counted, every row is, over \[τ₀, τ₁\] (D-42).
    // D-42
    pub fn bounds(&self) -> Aabb {
        let (rows, all) = self.counted_rows();
        let one = |i: usize| {
            let (a, b) = self
                .visible_interval(i, all)
                .unwrap_or((self.time.tau0, self.time.tau0));
            let pa = self.position_at(i, a);
            let pb = self.position_at(i, b);
            let margin = Vec3::splat(3.0 * Vec3::from_array(self.scale[i]).max_element());
            (pa.min(pb) - margin, pa.max(pb) + margin)
        };
        let empty = || (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let join = |(a0, a1): (Vec3, Vec3), (b0, b1): (Vec3, Vec3)| (a0.min(b0), a1.max(b1));
        let (min, max) = if rows.len() >= PARALLEL_ROWS {
            rows.par_iter().map(|&i| one(i)).reduce(empty, join)
        } else {
            rows.iter().map(|&i| one(i)).fold(empty(), join)
        };
        Aabb { min, max }
    }

    /// E10's content box B_c in model space: along each axis, from the nearest-rank 1st
    /// to the 99th percentile (the ⌈0.01 n⌉-th and ⌈0.99 n⌉-th smallest of n values) of
    /// the counted rows' positions μx at their visible intervals' midpoints, with no
    /// margin. The rows are `bounds`'s; in its fallback (no counted row) every row's
    /// interval is \[τ₀, τ₁\], so its midpoint is (τ₀ + τ₁)/2. `bounds` when B_c's longest
    /// side is 0 (a single counted row, or all of them at one point).
    // D-42
    pub fn content_bounds(&self) -> Aabb {
        let (rows, all) = self.counted_rows();
        let mid = |i: usize| {
            let (a, b) = self
                .visible_interval(i, all)
                .unwrap_or((self.time.tau0, self.time.tau0));
            self.position_at(i, 0.5 * (a + b))
        };
        let points: Vec<Vec3> = if rows.len() >= PARALLEL_ROWS {
            rows.par_iter().map(|&i| mid(i)).collect()
        } else {
            rows.iter().map(|&i| mid(i)).collect()
        };
        let n = points.len();
        // Nearest rank, 1-based, in integers: ⌈k n / 100⌉.
        let rank = |k: usize| (k * n).div_ceil(100).max(1);
        let (lo, hi) = (rank(1) - 1, rank(99) - 1);
        let mut min = Vec3::ZERO;
        let mut max = Vec3::ZERO;
        let mut axis: Vec<f32> = Vec::with_capacity(n);
        for a in 0..3 {
            axis.clear();
            axis.extend(points.iter().map(|p| p[a]));
            min[a] = *axis.select_nth_unstable_by(lo, f32::total_cmp).1;
            max[a] = *axis.select_nth_unstable_by(hi, f32::total_cmp).1;
        }
        let b = Aabb { min, max };
        if b.size().max_element() > 0.0 {
            b
        } else {
            self.bounds()
        }
    }

    /// E10's counted rows, and whether they are the fallback's every row (D-42).
    fn counted_rows(&self) -> (Vec<usize>, bool) {
        let rows: Vec<usize> = (0..self.len())
            .filter(|&i| self.visible_interval(i, false).is_some())
            .collect();
        if rows.is_empty() {
            ((0..self.len()).collect(), true)
        } else {
            (rows, false)
        }
    }

    /// Row `i`'s visible interval clipped to \[τ₀, τ₁\] (E10); `None` when it is not
    /// counted. With `all` (E10's fallback), \[τ₀, τ₁\] for every row.
    fn visible_interval(&self, i: usize, all: bool) -> Option<(f64, f64)> {
        let (tau0, tau1) = (self.time.tau0, self.time.tau1);
        if all {
            return Some((tau0, tau1));
        }
        let alpha_skip = self.conventions.alpha_skip;
        let opacity = self.opacity[i];
        if opacity <= alpha_skip {
            return None;
        }
        if self.is_single_timestamp() {
            return Some((tau0, tau0));
        }
        // E2's cull σ_eff(τ) ≥ α_skip, as E3's half-width without the index's widening.
        let ratio = f64::from(opacity) / f64::from(alpha_skip);
        let h = self.duration[i] * (2.0 * ratio.ln()).sqrt();
        let c = self.time_center[i];
        let (a, b) = ((c - h).max(tau0), (c + h).min(tau1));
        (a <= b).then_some((a, b))
    }

    /// Row `i`'s μx(τ) (E1, in float64, rounded once; D-42).
    fn position_at(&self, i: usize, tau: f64) -> Vec3 {
        moved_position_f64(self.position[i], self.velocity[i], self.time_center[i], tau)
    }

    /// §1.2's reader rules on the arrays in memory, reporting the first rule broken in
    /// §1.2's order: `sh_degree` in 0–3 (E-FILE-SH); τ₀ and τ₁ finite with τ₀ ≤ τ₁
    /// (E-FILE-VALUE); N ≥ 1 and the same N in every array (E-FILE-SHAPE), the `sh`
    /// length N·(L+1)²·3 (E-FILE-SH); every value finite; durations and scales > 0;
    /// rotation norms > 10⁻⁸; opacities in (0, 1) (E-FILE-VALUE); with a single timestamp
    /// every time center τ₀ and every velocity 0 (E-FILE-SINGLE, openftgs D-71); the
    /// conventions' numbers finite and the background in \[0, 1\] (E-FILE-VALUE, Q-8).
    /// The container rules (format, version, keys, dtypes) are the readers'.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.validate_rows(self.len() >= PARALLEL_ROWS)
    }

    /// `validate`, with the per-row checks in parallel when `parallel`. Each rule is
    /// checked over every row before the next, and a rule's search reports its lowest
    /// broken row, so both paths give the same error.
    fn validate_rows(&self, parallel: bool) -> Result<(), CoreError> {
        let l = self.sh_degree;
        if l > 3 {
            return Err(CoreError::Sh(format!("sh_degree {l}")));
        }
        let (tau0, tau1) = (self.time.tau0, self.time.tau1);
        if !tau0.is_finite() || !tau1.is_finite() || tau1 < tau0 {
            return Err(CoreError::Value(format!("tau0 = {tau0}, tau1 = {tau1}")));
        }

        let n = self.len();
        if n == 0 {
            return Err(CoreError::Shape(
                "position has shape [0, 3], expected N ≥ 1".to_owned(),
            ));
        }
        let lengths = [
            ("time_center", self.time_center.len()),
            ("duration", self.duration.len()),
            ("velocity", self.velocity.len()),
            ("scale", self.scale.len()),
            ("rotation", self.rotation.len()),
            ("opacity", self.opacity.len()),
        ];
        for (key, len) in lengths {
            if len != n {
                return Err(CoreError::Shape(format!(
                    "{key} has {len} rows, expected N = {n}"
                )));
            }
        }
        let k = (usize::from(l) + 1).pow(2);
        if self.sh.len() != n * k * 3 {
            return Err(CoreError::Sh(format!(
                "sh holds {} values, expected [N, {k}, 3] = {} for sh_degree {l}",
                self.sh.len(),
                n * k * 3
            )));
        }

        check_finite("position", self.position.as_flattened(), parallel)?;
        check_finite("time_center", &self.time_center, parallel)?;
        check_finite("duration", &self.duration, parallel)?;
        check_finite("velocity", self.velocity.as_flattened(), parallel)?;
        check_finite("scale", self.scale.as_flattened(), parallel)?;
        check_finite("rotation", self.rotation.as_flattened(), parallel)?;
        check_finite("opacity", &self.opacity, parallel)?;
        check_finite("sh", &self.sh, parallel)?;

        if let Some(i) = first_broken(&self.duration, parallel, |&d| d <= 0.0) {
            return Err(CoreError::Value(format!(
                "duration[{i}] = {}",
                self.duration[i]
            )));
        }
        if let Some(i) = first_broken(&self.scale, parallel, |s| s.iter().any(|&x| x <= 0.0)) {
            return Err(CoreError::Value(format!(
                "scale[{i}] = {:?}",
                self.scale[i]
            )));
        }
        if let Some(i) = first_broken(&self.rotation, parallel, |q| {
            q.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt() <= 1e-8
        }) {
            return Err(CoreError::Value(format!(
                "rotation[{i}] = {:?}",
                self.rotation[i]
            )));
        }
        if let Some(i) = first_broken(&self.opacity, parallel, |&o| o <= 0.0 || o >= 1.0) {
            return Err(CoreError::Value(format!(
                "opacity[{i}] = {}",
                self.opacity[i]
            )));
        }

        // openftgs D-71
        if self.is_single_timestamp() {
            if let Some(i) = first_broken(&self.time_center, parallel, |&t| t != tau0) {
                return Err(CoreError::SingleTimestamp(format!(
                    "time_center[{i}] = {} ≠ tau0 = {tau0}",
                    self.time_center[i]
                )));
            }
            if let Some(i) = first_broken(&self.velocity, parallel, |v| v.iter().any(|&x| x != 0.0))
            {
                return Err(CoreError::SingleTimestamp(format!(
                    "velocity[{i}] = {:?}",
                    self.velocity[i]
                )));
            }
        }

        self.validate_conventions()
    }

    /// The conventions block's rule (§1.2, openftgs D-84) and its numbers finite (Q-8).
    // Q-8
    fn validate_conventions(&self) -> Result<(), CoreError> {
        let c = &self.conventions;
        let numbers = [
            ("alpha_max", c.alpha_max),
            ("eps2d", c.eps2d),
            ("near", c.near),
            ("far", c.far),
            ("alpha_skip", c.alpha_skip),
            ("transmittance_stop", c.transmittance_stop.unwrap_or(0.0)),
            ("pixel_center_offset", c.pixel_center_offset),
            ("jacobian_clamp_margin", c.jacobian_clamp_margin),
            ("screen_extent_max", c.screen_extent_max),
        ];
        for (key, x) in numbers {
            if !x.is_finite() {
                return Err(CoreError::Value(format!("render_conventions.{key} = {x}")));
            }
        }
        if !c.background.iter().all(|x| (0.0..=1.0).contains(x)) {
            return Err(CoreError::Value(format!(
                "render_conventions.background = {:?}",
                c.background
            )));
        }
        Ok(())
    }

    /// §1.2 load option 1: with a cap L′ < L, drops the bands above L′ (coefficients
    /// (L′+1)² onward of every row) and sets L to L′; a cap at or above L is the identity.
    pub fn cap_sh(&mut self, degree: u8) {
        if degree >= self.sh_degree {
            return;
        }
        let row = (usize::from(self.sh_degree) + 1).pow(2) * 3;
        let kept = (usize::from(degree) + 1).pow(2) * 3;
        self.sh = self
            .sh
            .chunks_exact(row)
            .flat_map(|r| r[..kept].iter().copied())
            .collect();
        self.sh_degree = degree;
    }

    /// Whether τ₀ = τ₁ (D-9).
    pub fn is_single_timestamp(&self) -> bool {
        self.time.is_single()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid model of `n` rows, degree 0, over τ ∈ [0, 1].
    fn model(n: usize) -> Model {
        Model {
            position: vec![[0.0; 3]; n],
            time_center: vec![0.5; n],
            duration: vec![1.0; n],
            velocity: vec![[0.0; 3]; n],
            scale: vec![[1.0; 3]; n],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
            opacity: vec![0.5; n],
            sh: vec![0.0; n * 3],
            sh_degree: 0,
            time: TimeSpan {
                tau0: 0.0,
                tau1: 1.0,
                time_unit: "s".to_owned(),
            },
            conventions: RenderConventions::default(),
            producer: String::new(),
            identity: [0; 32],
            kind: ModelKind::ExchangeFile,
        }
    }

    /// Above the threshold, so `validate` takes the parallel path.
    const N: usize = PARALLEL_ROWS + 5_000;

    /// The parallel and sequential paths both give `expected` (§1.2's rule order first,
    /// then the lowest row), and `validate` gives it too.
    fn assert_both(m: &Model, expected: &str) {
        let parallel = format!("{:?}", m.validate_rows(true));
        let sequential = format!("{:?}", m.validate_rows(false));
        assert_eq!(parallel, expected, "parallel");
        assert_eq!(sequential, expected, "sequential");
        assert_eq!(format!("{:?}", m.validate()), expected, "validate");
    }

    #[test]
    fn validate_rows_in_parallel_reports_the_sequential_error() {
        assert_both(&model(N), "Ok(())");

        // Two rows break `duration`, later rows break `scale`, an earlier row breaks
        // `opacity`: the first rule broken is `duration`, at its lowest row.
        let mut m = model(N);
        m.opacity[1_000] = 2.0;
        m.scale[N - 10][1] = 0.0;
        m.duration[N - 20] = -1.0;
        m.duration[N - 30] = -2.0;
        assert_both(&m, &format!("Err(Value(\"duration[{}] = -2\"))", N - 30));

        // Non-finite values report the flat index: row r, component c is 3r + c.
        let mut m = model(N);
        m.time_center[100] = f64::NAN;
        m.position[N - 1][2] = f32::NAN;
        m.position[N - 2][0] = f32::INFINITY;
        assert_both(
            &m,
            &format!("Err(Value(\"position[{}] = inf\"))", 3 * (N - 2)),
        );

        // A single timestamp: `time_center` before `velocity`.
        let mut m = model(N);
        m.time.tau1 = 0.0;
        m.time_center.fill(0.0);
        m.velocity[N - 50] = [0.0, 1.0, 0.0];
        m.time_center[N - 40] = 0.5;
        m.time_center[N - 45] = 0.25;
        assert_both(
            &m,
            &format!(
                "Err(SingleTimestamp(\"time_center[{}] = 0.25 ≠ tau0 = 0\"))",
                N - 45
            ),
        );
    }
}
