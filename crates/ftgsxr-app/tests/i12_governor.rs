// SPDX-License-Identifier: Apache-2.0

//! I-12 (E16, `cpu`): injected `SplatStats` step `SplatTierState` in order, over the
//! desktop ladder (owner, 2026-11-02; Q-153): render scale 1.0 → 0.6, then the SH cap
//! 3 → 1, and back in the reverse order; f_budget at the 60 Hz fallback, no window.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{Governor, QualityTier, SettingsRes};
use ftgsxr_render::{FrameStats, GpuTimes, SplatStats, SplatTierState};
use support::{Setup, TestApp};

/// A statistics block whose GPU frame took `total_ms`.
fn block(total_ms: f32) -> FrameStats {
    FrameStats {
        visible_splats: 0,
        visible_centers: 0,
        capacity: 0,
        overflow: false,
        non_finite: 0,
        gpu_times_ms: GpuTimes {
            cull: 0.0,
            sort: 0.0,
            draw: total_ms,
            total: total_ms,
        },
    }
}

/// (render scale in tenths, SH cap).
fn knobs(app: &App) -> (i32, u8) {
    let s = app.world().resource::<SplatTierState>();
    ((s.render_scale * 10.0).round() as i32, s.sh_cap)
}

/// Runs `frames` frames, each with a fresh block of `total_ms`; the knobs after each
/// frame that changed them, with that frame's number (from 1).
fn run(t: &mut TestApp, total_ms: f32, frames: u32) -> Vec<(u32, (i32, u8))> {
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

#[test]
fn i12_splat_tier_state_steps_in_order_from_injected_frame_times() {
    let mut t = TestApp::new("i12_steps");
    assert!(t.app.world().resource::<Governor>().enabled);
    assert_eq!(knobs(&t.app), (10, 3));
    // 60 Hz: f_budget = 16.667 − 1.5 ms; 30 ms is e ≈ 2.
    let down = run(&mut t, 30.0, 100);
    assert_eq!(
        down,
        vec![
            (10, (9, 3)),
            (20, (8, 3)),
            (30, (7, 3)),
            (40, (6, 3)),
            (50, (6, 2)),
            (60, (6, 1)),
        ]
    );
    // Frames without a new block count to neither streak.
    t.frames(500);
    assert_eq!(knobs(&t.app), (6, 1));
    // 1 ms is e ≈ 0.07.
    let up = run(&mut t, 1.0, 120 * 7);
    assert_eq!(
        up,
        vec![
            (120, (6, 2)),
            (240, (6, 3)),
            (360, (7, 3)),
            (480, (8, 3)),
            (600, (9, 3)),
            (720, (10, 3)),
        ]
    );
}

#[test]
fn i12_the_governor_is_off_when_the_settings_turn_it_off() {
    let mut t = TestApp::with("i12_off", Setup::default(), |app| {
        let mut s = SettingsRes::default();
        s.0.general.governor = false;
        app.insert_resource(s);
    });
    assert!(run(&mut t, 30.0, 100).is_empty());
    assert!(!t.app.world().resource::<Governor>().enabled);
}

#[test]
fn i12_a_tier_chosen_by_hand_turns_the_governor_off() {
    // Q-179: a tier other than the build's default.
    let setup = Setup {
        tier: QualityTier::quest_low(),
        ..Setup::default()
    };
    let mut t = TestApp::with("i12_manual", setup, |_| {});
    assert!(run(&mut t, 30.0, 100).is_empty());
    assert!(!t.app.world().resource::<Governor>().enabled);
}
