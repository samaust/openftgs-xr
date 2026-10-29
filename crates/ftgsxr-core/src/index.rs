// SPDX-License-Identifier: Apache-2.0

//! The temporal index (§2.1 `ftgsxr_core::index`; E3, D-10): the row order
//! persistent-first, then transient rows by time center (§1.2 load option 5), and the
//! dispatch range a cull pass visits at τ, a superset of the active set. The cull pass
//! still decides activity exactly, so images never depend on the index (D-10).

use std::ops::Range;

use rayon::prelude::*;

use crate::Model;
use crate::motion::temporal_window;

/// The temporal index (E3, D-10): persistent rows first, then transient rows by time center.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalIndex {
    /// P, the number of persistent rows: new rows 0..P.
    persistent: u32,
    /// The transient rows' time centers in new-row order (new rows P..N), non-decreasing.
    centers: Vec<f64>,
    /// w_max, the widest widened window of a transient row; 0 when none has one.
    w_max: f64,
}

/// The rows a cull pass visits at one τ, as new-row indices (positions in the order
/// `TemporalIndex::build` returns): every persistent row, then the contiguous run of
/// transient rows whose time centers lie within w_max of τ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchRange {
    /// The persistent rows, 0..P.
    pub persistent: Range<u32>,
    /// The transient rows visited at τ, a run inside P..N.
    pub transient: Range<u32>,
}

// §2.1 gives `len` without `is_empty`; a validated model is never empty (N ≥ 1).
#[allow(clippy::len_without_is_empty)]
impl TemporalIndex {
    /// Builds the index and returns the row permutation (new row → file row).
    ///
    /// A row is persistent when its E3 window \[time_center − w, time_center + w\], with w
    /// unwidened, covers \[τ₀, τ₁\]; persistent rows keep their file order. The other rows
    /// are transient, sorted by time center, ties in file order; w_max is the widest of
    /// their windows widened with openftgs's margins (`temporal_window`, D-10). A row with
    /// opacity ≤ α_skip has no window: it is transient and adds nothing to w_max. With a
    /// single timestamp σt ≡ 1 (D-9), so no row's activity depends on τ: every row is
    /// persistent and the permutation is the identity.
    ///
    /// # Panics
    ///
    /// When the model has more than `u32::MAX` rows.
    // Q-T9.1: rows with opacity ≤ α_skip, and the single-timestamp model.
    // Q-T9.2: unwidened w for persistence, file order within ties, w_max = 0 with no window.
    pub fn build(model: &Model) -> (TemporalIndex, Vec<u32>) {
        let n = u32::try_from(model.len()).expect("a temporal index holds at most u32::MAX rows");
        if model.time.is_single() {
            let index = TemporalIndex {
                persistent: n,
                centers: Vec::new(),
                w_max: 0.0,
            };
            return (index, (0..n).collect());
        }

        let (tau0, tau1, span) = (model.time.tau0, model.time.tau1, model.time.span());
        let alpha_skip = model.conventions.alpha_skip;
        let window = |i: usize, widened: bool| {
            temporal_window(
                model.opacity[i],
                model.duration[i],
                alpha_skip,
                span,
                widened,
            )
        };
        let persistent: Vec<bool> = (0..model.len())
            .into_par_iter()
            .map(|i| {
                let tc = model.time_center[i];
                window(i, false).is_some_and(|w| tc - tau0 <= w && tau1 - tc <= w)
            })
            .collect();

        let mut order: Vec<u32> = (0..n).filter(|&i| persistent[i as usize]).collect();
        let p = order.len();
        let mut transient: Vec<u32> = (0..n).filter(|&i| !persistent[i as usize]).collect();
        transient.par_sort_unstable_by(|&a, &b| {
            let (ta, tb) = (model.time_center[a as usize], model.time_center[b as usize]);
            ta.total_cmp(&tb).then(a.cmp(&b))
        });
        let w_max = transient
            .par_iter()
            .filter_map(|&i| window(i as usize, true))
            .reduce(|| 0.0, f64::max);
        let centers = transient
            .iter()
            .map(|&i| model.time_center[i as usize])
            .collect();
        order.extend(transient);

        let index = TemporalIndex {
            persistent: u32::try_from(p).expect("P ≤ N"),
            centers,
            w_max,
        };
        (index, order)
    }

    /// The rows to visit at τ: every persistent row and the transient rows whose time
    /// center c satisfies |τ − c| ≤ w_max in float64, a superset of the active set (U-8).
    /// A τ that is not finite visits the persistent rows alone.
    pub fn range(&self, tau: f64) -> DispatchRange {
        let w = self.w_max;
        // Both predicates are monotone in c, since float64 subtraction is.
        let start = self.centers.partition_point(|&c| c < tau && tau - c > w);
        let end = self.centers.partition_point(|&c| c <= tau || c - tau <= w);
        let p = self.persistent;
        let at = |k: usize| p + u32::try_from(k).expect("N ≤ u32::MAX");
        DispatchRange {
            persistent: 0..p,
            transient: at(start)..at(end.max(start)),
        }
    }

    /// N, the number of rows indexed.
    pub fn len(&self) -> u32 {
        self.persistent + u32::try_from(self.centers.len()).expect("N ≤ u32::MAX")
    }
}
