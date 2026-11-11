// SPDX-License-Identifier: Apache-2.0

//! The governor on the headset (E16, D-47, §1.12, `cpu`; #723, Q-404): while `XrDisplay`
//! exists the system steps `HEADSET_LADDER` against f_budget from `XrDisplay.refresh_hz`,
//! here written by the tests as `ftgsxr_xr`'s `display` would; without it, Q-153's
//! desktop rule (I-12).

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{Governor, XrDisplay};
use ftgsxr_render::{Foveation, FrameStats, GpuTimes, SortMode, SplatStats, SplatTierState};
use support::{Setup, TestApp};

/// A statistics block whose GPU frame took `total_ms`.
fn block(total_ms: f32) -> FrameStats {
    FrameStats {
        visible_splats: 0,
        visible_centers: 0,
        capacity: 0,
        overflow: false,
        non_finite: 0,
        trimmed: 0,
        gpu_times_ms: GpuTimes {
            cull: 0.0,
            sort: 0.0,
            draw: total_ms,
            total: total_ms,
        },
    }
}

/// (render scale in tenths, foveation, SH cap, sort mode).
type Knobs = (i32, Foveation, u8, SortMode);

fn knobs(app: &App) -> Knobs {
    let s = app.world().resource::<SplatTierState>();
    (
        (s.render_scale * 10.0).round() as i32,
        s.foveation,
        s.sh_cap,
        s.sort_mode,
    )
}

/// The app on the desktop tier (the build's default, so the governor is on, Q-179) with
/// `display` inserted before `AppPlugin`, as `XrPlugin` inserts it.
#[track_caller]
fn headset(name: &str, display: XrDisplay) -> TestApp {
    TestApp::with(name, Setup::default(), |app| {
        app.insert_resource(display);
    })
}

/// An `XrDisplay` at `hz` that offers foveation.
fn at(hz: f32) -> XrDisplay {
    XrDisplay {
        refresh_hz: Some(hz),
        available_rates: vec![72.0, 90.0],
        foveation: true,
        ..XrDisplay::default()
    }
}

/// Runs `frames` frames, each with a fresh block of `total_ms`; the knobs after each
/// frame that changed them, with that frame's number (from 1).
fn run(t: &mut TestApp, total_ms: f32, frames: u32) -> Vec<(u32, Knobs)> {
    let mut seen = Vec::new();
    let mut last = knobs(&t.app);
    for f in 1..=frames {
        t.app.world_mut().resource_mut::<SplatStats>().0 = Some(block(total_ms));
        t.app.update();
        let now = knobs(&t.app);
        if now != last {
            seen.push((f, now));
            last = now;
        }
    }
    seen
}

// Q-404
#[test]
fn the_headset_budget_follows_the_display_refresh_rate() {
    // E16 by hand: at 72 Hz f_budget = 1000/72 − 1.5 = 12.39 ms, so e > 1.05 above
    // 13.01 ms; at 90 Hz 9.61 ms, over above 10.09 ms. The desktop's 60 Hz fallback
    // (15.17 ms, over above 15.93 ms) would step on none of these.
    for (hz, in_band, over) in [(72.0, 12.9, 13.1), (90.0, 10.0, 10.2)] {
        let mut t = headset(&format!("gov-headset-budget-{hz}"), at(hz));
        assert!(t.app.world().resource::<Governor>().enabled);
        assert!(run(&mut t, in_band, 50).is_empty(), "{hz} Hz, {in_band} ms");
        assert_eq!(
            run(&mut t, over, 10),
            vec![(10, (9, Foveation::Off, 3, SortMode::PerView))],
            "{hz} Hz, {over} ms"
        );
    }
}

/// E16's step-down order from the desktop tier's knobs (1.0, off, 3, per view): render
/// scale ×4, foveation ×3, SH cap ×2, sort mode ×1.
const FULL_DOWN: [Knobs; 10] = [
    (9, Foveation::Off, 3, SortMode::PerView),
    (8, Foveation::Off, 3, SortMode::PerView),
    (7, Foveation::Off, 3, SortMode::PerView),
    (6, Foveation::Off, 3, SortMode::PerView),
    (6, Foveation::Low, 3, SortMode::PerView),
    (6, Foveation::Medium, 3, SortMode::PerView),
    (6, Foveation::High, 3, SortMode::PerView),
    (6, Foveation::High, 2, SortMode::PerView),
    (6, Foveation::High, 1, SortMode::PerView),
    (6, Foveation::High, 1, SortMode::SharedCenter),
];

/// `expected` as `run` reports it, one change each `every` frames.
fn each(every: u32, expected: &[Knobs]) -> Vec<(u32, Knobs)> {
    expected
        .iter()
        .enumerate()
        .map(|(i, k)| (every * (i as u32 + 1), *k))
        .collect()
}

// Q-404
#[test]
fn injected_frame_times_step_the_headset_ladder_in_e16s_order() {
    let mut t = headset("gov-headset-ladder", at(72.0));
    let start = knobs(&t.app);
    // 30 ms at 72 Hz is e ≈ 2.4.
    assert_eq!(run(&mut t, 30.0, 150), each(10, &FULL_DOWN));
    // Frames without a new block count to neither streak.
    t.frames(500);
    // 1 ms is e ≈ 0.08: released in the reverse order, back to the tier's values.
    let mut up: Vec<Knobs> = FULL_DOWN.iter().rev().skip(1).copied().collect();
    up.push(start);
    assert_eq!(run(&mut t, 1.0, 120 * 12), each(120, &up));
}

/// The ladder without its foveation knob, from the desktop tier's knobs: render scale ×4,
/// SH cap ×2, sort mode ×1, the foveation left at the tier's `off`.
const NO_FOVEATION_DOWN: [Knobs; 7] = [
    (9, Foveation::Off, 3, SortMode::PerView),
    (8, Foveation::Off, 3, SortMode::PerView),
    (7, Foveation::Off, 3, SortMode::PerView),
    (6, Foveation::Off, 3, SortMode::PerView),
    (6, Foveation::Off, 2, SortMode::PerView),
    (6, Foveation::Off, 1, SortMode::PerView),
    (6, Foveation::Off, 1, SortMode::SharedCenter),
];

/// Steps down and back up; both ways skip the foveation.
fn skips_the_foveation(t: &mut TestApp) {
    let start = knobs(&t.app);
    assert_eq!(run(t, 30.0, 120), each(10, &NO_FOVEATION_DOWN));
    let mut up: Vec<Knobs> = NO_FOVEATION_DOWN.iter().rev().skip(1).copied().collect();
    up.push(start);
    assert_eq!(run(t, 1.0, 120 * 9), each(120, &up));
}

// Q-404
#[test]
fn an_explicit_xr_foveation_takes_the_knob_off_the_ladder() {
    let mut t = TestApp::with("gov-headset-pinned", Setup::default(), |app| {
        app.insert_resource(at(72.0));
        let mut s = ftgsxr_app::SettingsRes::default();
        s.0.xr.foveation = Some("medium".into());
        app.insert_resource(s);
    });
    skips_the_foveation(&mut t);
}

// Q-404
#[test]
fn a_runtime_without_foveation_skips_the_knob() {
    let mut t = headset(
        "gov-headset-no-foveation",
        XrDisplay {
            foveation: false,
            ..at(72.0)
        },
    );
    skips_the_foveation(&mut t);
}

// Q-404
#[test]
fn no_frame_counts_while_no_session_gives_a_rate() {
    let mut t = headset(
        "gov-headset-no-rate",
        XrDisplay {
            refresh_hz: None,
            ..at(72.0)
        },
    );
    let start = knobs(&t.app);
    // e would be ≈ 2.4 at any of the Quest's rates and ≈ 2 at the desktop's fallback.
    assert!(run(&mut t, 30.0, 100).is_empty());
    // The rate arrives: the frames before it counted to no streak, so the first step
    // comes at the tenth frame after.
    t.app.world_mut().resource_mut::<XrDisplay>().refresh_hz = Some(72.0);
    assert_eq!(run(&mut t, 30.0, 10), each(10, &FULL_DOWN[..1]));
    assert_ne!(knobs(&t.app), start);
}

// Q-153, Q-404
#[test]
fn without_xr_display_the_desktop_ladder_and_budget_apply() {
    let mut t = TestApp::new("gov-desktop-budget");
    // Over at 72 Hz, in the band at the 60 Hz fallback (15.17 ms, over above 15.93 ms).
    assert!(run(&mut t, 13.1, 50).is_empty());
    // Over the fallback: the desktop ladder, render scale then the SH cap (I-12).
    let down = run(&mut t, 16.5, 100);
    assert_eq!(down.len(), 6, "{down:?}");
    assert!(
        down.iter()
            .all(|(_, k)| k.1 == Foveation::Off && k.3 == SortMode::PerView)
    );
}

// Q-416
#[test]
fn the_budget_override_counts_frames_without_a_rate() {
    let mut t = headset(
        "gov-headset-override",
        XrDisplay {
            refresh_hz: None,
            ..at(72.0)
        },
    );
    // R-17's override: 1 ms against 0.1 ms is e = 10.
    t.app
        .world_mut()
        .resource_mut::<Governor>()
        .budget_override_ms = Some(0.1);
    assert_eq!(run(&mut t, 1.0, 10), each(10, &FULL_DOWN[..1]));
}
