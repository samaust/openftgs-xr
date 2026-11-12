// SPDX-License-Identifier: Apache-2.0

//! U-22 (E16, D-47, `cpu`): a frame-time sequence drives the knobs down in the stated
//! order after 10 frames over and up after 120 under, over E16's full ladder (Q-153); a
//! manual tier disables the governor.

use crate::support;

use ftgsxr_app::{Governor, GovernorCounters, HEADSET_LADDER, Knob, QualityTier, govern};
use ftgsxr_render::{Foveation, SortMode, SplatTierState};

/// The desktop tier's knobs: every one of E16's knobs has room to step from them.
fn desktop_state() -> SplatTierState {
    let tier = QualityTier::desktop();
    SplatTierState {
        render_scale: tier.render_scale,
        sh_cap: 3,
        sort_mode: tier.sort_mode,
        foveation: tier.foveation,
        tier,
    }
}

/// The knobs as a tuple, render scale in tenths.
fn knobs(s: &SplatTierState) -> (i32, Foveation, u8, SortMode) {
    (
        (s.render_scale * 10.0).round() as i32,
        s.foveation,
        s.sh_cap,
        s.sort_mode,
    )
}

/// Feeds `e` for `frames` frames; every step taken, with the frame (from 1) it came on.
fn feed(
    gov: &Governor,
    c: &mut GovernorCounters,
    s: &mut SplatTierState,
    e: f32,
    frames: u32,
) -> Vec<(u32, Knob, bool)> {
    (1..=frames)
        .filter_map(|f| govern(gov, c, s, &HEADSET_LADDER, e).map(|(k, d)| (f, k, d)))
        .collect()
}

#[test]
fn u22_the_knobs_step_down_after_10_frames_over_and_up_after_120_under_in_order() {
    let gov = Governor::default();
    assert_eq!(
        (gov.enabled, gov.down_frames, gov.up_frames),
        (true, 10, 120)
    );
    let mut c = GovernorCounters::default();
    let mut s = desktop_state();
    let start = knobs(&s);

    // Nine frames over, then one in the band: no step, and the streak restarts.
    assert!(feed(&gov, &mut c, &mut s, 1.06, 9).is_empty());
    assert!(feed(&gov, &mut c, &mut s, 1.0, 1).is_empty());
    assert!(feed(&gov, &mut c, &mut s, 1.06, 9).is_empty());
    assert_eq!(knobs(&s), start);

    // e = 1.05 is not over (E16's strict inequality).
    assert!(feed(&gov, &mut c, &mut s, 1.05, 30).is_empty());

    // Down: render scale 1.0 → 0.6 (4), foveation off → high (3), SH cap 3 → 1 (2),
    // sort mode per view → shared center (1), one step each 10 frames.
    let down = feed(&gov, &mut c, &mut s, 2.0, 10 * 10 + 50);
    let order = [
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::Foveation,
        Knob::Foveation,
        Knob::Foveation,
        Knob::ShCap,
        Knob::ShCap,
        Knob::SortMode,
    ];
    assert_eq!(
        down,
        order
            .iter()
            .enumerate()
            .map(|(i, k)| (10 * (i as u32 + 1), *k, true))
            .collect::<Vec<_>>()
    );
    assert_eq!(knobs(&s), (6, Foveation::High, 1, SortMode::SharedCenter));

    // e = 0.8 is not under.
    assert!(feed(&gov, &mut c, &mut s, 0.8, 300).is_empty());
    // 119 under, one over: no step, the streak restarts.
    assert!(feed(&gov, &mut c, &mut s, 0.5, 119).is_empty());
    assert!(feed(&gov, &mut c, &mut s, 1.1, 1).is_empty());

    // Up in the reverse order, one step each 120 frames, back to the tier's values.
    let up = feed(&gov, &mut c, &mut s, 0.1, 120 * 10 + 500);
    assert_eq!(
        up,
        order
            .iter()
            .rev()
            .enumerate()
            .map(|(i, k)| (120 * (i as u32 + 1), *k, false))
            .collect::<Vec<_>>()
    );
    assert_eq!(knobs(&s), start);
    assert_eq!(s.render_scale, 1.0);
}

#[test]
fn u22_a_manual_tier_disables_the_governor() {
    let gov = Governor {
        enabled: false,
        ..Governor::default()
    };
    let mut c = GovernorCounters::default();
    let mut s = desktop_state();
    let before = s.clone();
    assert!(feed(&gov, &mut c, &mut s, 5.0, 500).is_empty());
    assert!(feed(&gov, &mut c, &mut s, 0.1, 500).is_empty());
    assert_eq!(s, before);
}

#[test]
fn u22_a_knob_is_released_only_to_the_tiers_own_value() {
    // quest-low: render scale 0.8, foveation high, SH cap 1, shared center.
    let tier = QualityTier::quest_low();
    let mut s = SplatTierState {
        render_scale: tier.render_scale,
        sh_cap: tier.layout.sh_cap.unwrap_or(3),
        sort_mode: tier.sort_mode,
        foveation: tier.foveation,
        tier,
    };
    let gov = Governor::default();
    let mut c = GovernorCounters::default();
    let down = feed(&gov, &mut c, &mut s, 2.0, 100);
    assert_eq!(
        down,
        vec![(10, Knob::RenderScale, true), (20, Knob::RenderScale, true)]
    );
    let up = feed(&gov, &mut c, &mut s, 0.1, 1000);
    assert_eq!(
        up,
        vec![
            (120, Knob::RenderScale, false),
            (240, Knob::RenderScale, false)
        ]
    );
    assert_eq!(knobs(&s), (8, Foveation::High, 1, SortMode::SharedCenter));
}

// Q-179
#[test]
fn u22_choosing_a_tier_by_hand_disables_the_governor() {
    // §1.12: "a tier chosen by hand turns the governor off": the app on the build's
    // default tier governs; on another tier (`--tier`, the settings) it does not.
    let enabled = |name: &str, tier: &str| {
        let mut t = support::TestApp::with(
            name,
            support::Setup {
                tier: QualityTier::by_name(tier).unwrap(),
                ..support::Setup::default()
            },
            |_| {},
        );
        t.frames(2);
        t.app.world().resource::<Governor>().enabled
    };
    assert!(enabled("u22-default-tier", "desktop"));
    assert!(!enabled("u22-manual-tier", "quest-low"));
}

// Q-178 (U-22's extension, #263, #723)
#[test]
fn u22_dynamic_foveation_is_left_alone_on_the_headset_ladder() {
    let mut s = desktop_state();
    s.foveation = Foveation::Dynamic;
    let gov = Governor::default();
    let mut c = GovernorCounters::default();
    // Down: the render scale (4), then straight to the SH cap (2) and the sort mode (1).
    let down = feed(&gov, &mut c, &mut s, 2.0, 100);
    let order = [
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::RenderScale,
        Knob::ShCap,
        Knob::ShCap,
        Knob::SortMode,
    ];
    assert_eq!(
        down,
        order
            .iter()
            .enumerate()
            .map(|(i, k)| (10 * (i as u32 + 1), *k, true))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        knobs(&s),
        (6, Foveation::Dynamic, 1, SortMode::SharedCenter)
    );
    // Up: the reverse, `Dynamic` unchanged.
    let up = feed(&gov, &mut c, &mut s, 0.1, 120 * 10);
    assert_eq!(
        up,
        order
            .iter()
            .rev()
            .enumerate()
            .map(|(i, k)| (120 * (i as u32 + 1), *k, false))
            .collect::<Vec<_>>()
    );
    assert_eq!(knobs(&s), (10, Foveation::Dynamic, 3, SortMode::PerView));
}
