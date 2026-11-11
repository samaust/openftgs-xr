// SPDX-License-Identifier: Apache-2.0

//! The runtime's performance counters (XR\_META\_performance\_metrics; §1.13 HUD,
//! Appendix E.1), read for the device spikes of #725 only: nothing else reads them yet.
//! Called through the openxr crate's raw function table as `display` calls its
//! extensions (Q-124), `unsafe` with SAFETY comments (§2.6).
//!
//! Inert until an entity whose name starts with [`PREFIX`] exists (Q-308's pattern): the
//! quest runner spawns `ftgsxr-test: metrics <label>` for a window and despawns it at
//! the window's end. While one exists the session's metrics system is enabled (once per
//! session; the extension says a query before it fails) and every counter the runtime
//! enumerates is queried each frame; at the window's end one line per counter is logged,
//! `perf metrics <label>: <path> unit <unit> samples <n> valid <v> changes <c> distinct
//! <d> median <m> min <a> max <b>`, after a header with the window's frames, seconds and
//! the running tier's knobs. Without the extension the window logs
//! `perf metrics <label>: XR_META_performance_metrics absent`. The counter paths are
//! the runtime's list, not the specification's ("may be provided", source log 419).
// Q-420

use std::time::Instant;

use bevy::app::{App, Update};
use bevy::log::info;
use bevy::prelude::{Name, Query, Res, ResMut, Resource};
use bevy_mod_openxr::resources::OxrInstance;
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::{XrPreDestroySession, XrState};
use ftgsxr_render::SplatTierState;
use openxr::sys;

use crate::XrError;
use crate::error::log_error;

/// The name prefix of the quest runner's window entities (Q-308).
pub(crate) const PREFIX: &str = "ftgsxr-test: metrics ";

/// One enumerated counter and its window's samples.
struct Counter {
    path: sys::Path,
    name: String,
    unit: Option<sys::PerformanceMetricsCounterUnitMETA>,
    values: Vec<Option<f64>>,
}

/// A window between the marker's arrival and its removal.
struct Window {
    label: String,
    started: Instant,
    frames: u32,
}

#[derive(Resource, Default)]
struct Metrics {
    /// Whether this session's metrics system was enabled (or the attempt failed).
    tried: bool,
    enabled: bool,
    counters: Vec<Counter>,
    window: Option<Window>,
}

/// Adds the systems (`display`'s plugin).
pub(crate) fn add(app: &mut App) {
    app.init_resource::<Metrics>()
        .add_systems(Update, run)
        .add_systems(XrPreDestroySession, session_ends);
}

/// The metrics system is per session (the extension's xrSetPerformanceMetricsStateMETA).
fn session_ends(mut m: ResMut<Metrics>) {
    m.tried = false;
    m.enabled = false;
}

fn run(
    names: Query<&Name>,
    mut m: ResMut<Metrics>,
    instance: Option<Res<OxrInstance>>,
    session: Option<Res<OxrSession>>,
    state: Option<Res<XrState>>,
    tier: Option<Res<SplatTierState>>,
) {
    let label = names
        .iter()
        .find_map(|n| n.as_str().strip_prefix(PREFIX).map(str::to_owned));
    let m = &mut *m;
    let current = m.window.as_ref().map(|w| w.label.clone());
    if current.is_some() && current != label {
        let w = m.window.take().expect("a window");
        finish(&w, &mut m.counters, tier.as_deref());
    }
    match label {
        None => return,
        Some(l) if m.window.is_none() => m.window = Some(start(l)),
        Some(_) => {}
    }
    let (Some(instance), Some(session)) = (instance, session) else {
        return;
    };
    let Some(fp) = instance.exts().meta_performance_metrics else {
        return;
    };
    if !m.tried {
        m.tried = true;
        m.enabled = enable(&instance, &session, &fp, &mut m.counters);
    }
    if !m.enabled || state.as_deref() != Some(&XrState::Running) {
        return;
    }
    if let Some(w) = m.window.as_mut() {
        w.frames += 1;
    }
    for c in &mut m.counters {
        let mut out = sys::PerformanceMetricsCounterMETA {
            ty: sys::PerformanceMetricsCounterMETA::TYPE,
            next: std::ptr::null(),
            counter_flags: sys::PerformanceMetricsCounterFlagsMETA::EMPTY,
            counter_unit: sys::PerformanceMetricsCounterUnitMETA::GENERIC,
            uint_value: 0,
            float_value: 0.0,
        };
        // SAFETY: the extension's table is loaded (checked above, so the pointer is the
        // runtime's); the session handle is alive (a main-world resource while the
        // session exists); the path is one the runtime enumerated; `out` is a fully
        // initialised output struct with its TYPE set, outliving the call.
        let res =
            unsafe { (fp.query_performance_metrics_counter)(session.as_raw(), c.path, &mut out) };
        if res.into_raw() < 0 {
            c.values.push(None);
        } else {
            c.unit = Some(out.counter_unit);
            c.values.push(value(&out));
        }
    }
}

fn start(label: String) -> Window {
    info!("perf metrics {label}: window started");
    Window {
        label,
        started: Instant::now(),
        frames: 0,
    }
}

/// Enumerates the runtime's counters (the two-call idiom) and enables the metrics system
/// for the session; logs the list. `false` when a call fails.
fn enable(
    instance: &OxrInstance,
    session: &OxrSession,
    fp: &openxr::raw::PerformanceMetricsMETA,
    counters: &mut Vec<Counter>,
) -> bool {
    let mut n = 0u32;
    // SAFETY: the table is loaded; the instance handle is alive (a resource for the app's
    // lifetime); a capacity of 0 with a null array asks for the count, as the extension
    // defines, and `n` outlives the call.
    let res = unsafe {
        (fp.enumerate_performance_metrics_counter_paths)(
            instance.as_raw(),
            0,
            &mut n,
            std::ptr::null_mut(),
        )
    };
    if res.into_raw() < 0 {
        log_error(&XrError::Runtime(format!(
            "performance metrics paths: {res:?}"
        )));
        return false;
    }
    let mut paths = vec![sys::Path::NULL; n as usize];
    // SAFETY: as above; `paths` holds `n` elements, the capacity passed, and outlives
    // the call.
    let res = unsafe {
        (fp.enumerate_performance_metrics_counter_paths)(
            instance.as_raw(),
            n,
            &mut n,
            paths.as_mut_ptr(),
        )
    };
    if res.into_raw() < 0 {
        log_error(&XrError::Runtime(format!(
            "performance metrics paths: {res:?}"
        )));
        return false;
    }
    paths.truncate(n as usize);
    counters.clear();
    for path in paths {
        let name = instance
            .path_to_string(path)
            .unwrap_or_else(|e| format!("<path {}: {e:?}>", path.into_raw()));
        counters.push(Counter {
            path,
            name,
            unit: None,
            values: Vec::new(),
        });
    }
    info!(
        "perf metrics: {} counters: {}",
        counters.len(),
        counters
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let state = sys::PerformanceMetricsStateMETA {
        ty: sys::PerformanceMetricsStateMETA::TYPE,
        next: std::ptr::null(),
        enabled: true.into(),
    };
    // SAFETY: the table is loaded; the session handle is alive (above); `state` is fully
    // initialised with its TYPE and outlives the call.
    let res = unsafe { (fp.set_performance_metrics_state)(session.as_raw(), &state) };
    if res.into_raw() < 0 {
        log_error(&XrError::Runtime(format!(
            "enabling the performance metrics: {res:?}"
        )));
        return false;
    }
    info!("perf metrics: enabled for the session");
    true
}

/// Logs the window's lines and clears its samples.
fn finish(w: &Window, counters: &mut [Counter], tier: Option<&SplatTierState>) {
    let knobs = tier.map_or_else(
        || "no tier".to_owned(),
        |t| {
            format!(
                "tier {} render scale {} sh cap {} sort {:?} tier foveation {:?}",
                t.tier.name, t.render_scale, t.sh_cap, t.sort_mode, t.foveation
            )
        },
    );
    info!(
        "perf metrics {}: {} frames over {:.2} s, {knobs}",
        w.label,
        w.frames,
        w.started.elapsed().as_secs_f64()
    );
    if counters.is_empty() {
        info!(
            "perf metrics {}: XR_META_performance_metrics absent or not enabled",
            w.label
        );
    }
    for c in counters {
        let s = summarize(&c.values);
        info!(
            "perf metrics {}: {} unit {} samples {} valid {} changes {} distinct {} median {} min {} max {}",
            w.label,
            c.name,
            c.unit.map_or("unknown", unit_name),
            s.samples,
            s.valid,
            s.changes,
            s.distinct,
            fmt(s.median),
            fmt(s.min),
            fmt(s.max)
        );
        c.values.clear();
    }
}

fn fmt(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |v| format!("{v}"))
}

/// A counter's value by its flags: the float value where valid, else the integer value,
/// else none.
fn value(c: &sys::PerformanceMetricsCounterMETA) -> Option<f64> {
    use sys::PerformanceMetricsCounterFlagsMETA as F;
    if c.counter_flags.contains(F::FLOAT_VALUE_VALID) {
        Some(f64::from(c.float_value))
    } else if c.counter_flags.contains(F::UINT_VALUE_VALID) {
        Some(f64::from(c.uint_value))
    } else {
        None
    }
}

/// The unit's name (source log 419).
fn unit_name(u: sys::PerformanceMetricsCounterUnitMETA) -> &'static str {
    use sys::PerformanceMetricsCounterUnitMETA as U;
    match u {
        U::GENERIC => "generic",
        U::PERCENTAGE => "%",
        U::MILLISECONDS => "ms",
        U::BYTES => "bytes",
        U::HERTZ => "Hz",
        _ => "unknown",
    }
}

/// A window's samples of one counter.
#[derive(Debug, PartialEq)]
struct Summary {
    samples: usize,
    valid: usize,
    /// Consecutive valid samples that differ.
    changes: usize,
    distinct: usize,
    /// The upper median of the valid samples (the middle one, or the upper of the two).
    median: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

fn summarize(values: &[Option<f64>]) -> Summary {
    let valid: Vec<f64> = values.iter().flatten().copied().collect();
    let changes = valid.windows(2).filter(|w| w[0] != w[1]).count();
    let mut sorted = valid.clone();
    sorted.sort_by(f64::total_cmp);
    let mut distinct = sorted.clone();
    distinct.dedup();
    Summary {
        samples: values.len(),
        valid: valid.len(),
        changes,
        distinct: distinct.len(),
        median: sorted.get(sorted.len() / 2).copied(),
        min: sorted.first().copied(),
        max: sorted.last().copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter(flags: u64, uint: u32, float: f32) -> sys::PerformanceMetricsCounterMETA {
        sys::PerformanceMetricsCounterMETA {
            ty: sys::PerformanceMetricsCounterMETA::TYPE,
            next: std::ptr::null(),
            counter_flags: sys::PerformanceMetricsCounterFlagsMETA::from_raw(flags),
            counter_unit: sys::PerformanceMetricsCounterUnitMETA::MILLISECONDS,
            uint_value: uint,
            float_value: float,
        }
    }

    /// The flags say which value is valid (source log 419: bit 1 uint, bit 2 float).
    #[test]
    fn value_follows_the_counters_flags() {
        assert_eq!(value(&counter(0, 7, 2.5)), None);
        assert_eq!(value(&counter(0b001, 7, 2.5)), None);
        assert_eq!(value(&counter(0b011, 7, 2.5)), Some(7.0));
        assert_eq!(value(&counter(0b101, 7, 2.5)), Some(2.5));
    }

    #[test]
    fn unit_name_names_the_extensions_units() {
        use sys::PerformanceMetricsCounterUnitMETA as U;
        assert_eq!(unit_name(U::MILLISECONDS), "ms");
        assert_eq!(unit_name(U::PERCENTAGE), "%");
        assert_eq!(unit_name(U::HERTZ), "Hz");
        assert_eq!(unit_name(U::BYTES), "bytes");
        assert_eq!(unit_name(U::GENERIC), "generic");
    }

    /// Five frames of a counter that updates twice, one failed query: 4 valid samples,
    /// 2 changes, 3 distinct values, the upper median 2.0.
    #[test]
    fn summarize_counts_changes_and_distinct_values() {
        let s = summarize(&[Some(1.0), Some(1.0), None, Some(2.0), Some(3.0)]);
        assert_eq!(
            s,
            Summary {
                samples: 5,
                valid: 4,
                changes: 2,
                distinct: 3,
                median: Some(2.0),
                min: Some(1.0),
                max: Some(3.0),
            }
        );
        let empty = summarize(&[None, None]);
        assert_eq!((empty.valid, empty.median, empty.distinct), (0, None, 0));
    }
}
