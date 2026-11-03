// SPDX-License-Identifier: Apache-2.0

//! The live tier switch (#373; §1.7 Panels, §1.12, D-37; Q-179, Q-229, Q-230, Q-231): a
//! change of `[general] tier` while the app runs switches the running tier, the governor
//! follows Q-179's rule, and the loaded instances are reloaded with the new tier's layout
//! on their own entities, without a restart.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{Governor, QualityTier, SettingsRes};
use ftgsxr_render::SplatTierState;
use support::{Setup, TestApp};

/// The running tier's name.
fn running(app: &App) -> String {
    app.world().resource::<SplatTierState>().tier.name.clone()
}

/// Sets `[general] tier` as the Display or Settings tab's choice does.
fn pick(app: &mut App, tier: &str) {
    app.world_mut().resource_mut::<SettingsRes>().0.general.tier = tier.into();
}

// Q-231, Q-179
#[test]
fn live_tier_switch_follows_the_settings() {
    let mut t = TestApp::new("live_follow");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
    assert!(t.app.world().resource::<Governor>().enabled);
    pick(&mut t.app, "quest-low");
    t.frames(2);
    // §1.12's table: `quest-low` runs at render scale 0.8, SH cap 1; the knobs start at
    // the new tier's own values.
    let s = t.app.world().resource::<SplatTierState>().clone();
    assert_eq!(s.tier, QualityTier::quest_low());
    assert_eq!((s.render_scale, s.sh_cap), (0.8, 1));
    // Q-179: a tier chosen by hand turns the governor off.
    assert!(!t.app.world().resource::<Governor>().enabled);
    // Back to the build's default with `[general] governor` on: back on.
    pick(&mut t.app, "desktop");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
    assert!(t.app.world().resource::<Governor>().enabled);
    // A name no tier has leaves the running tier alone.
    pick(&mut t.app, "no-such-tier");
    t.frames(2);
    assert_eq!(running(&t.app), "desktop");
}

// Q-231: `--tier` at startup holds until the user picks a tier.
#[test]
fn live_tier_switch_leaves_a_startup_override_alone() {
    let setup = Setup {
        tier: QualityTier::quest_high(),
        ..Setup::default()
    };
    let mut t = TestApp::with("live_override", setup, |_| {});
    // The settings say `desktop` (the default); the session runs `quest-high`.
    t.frames(5);
    assert_eq!(running(&t.app), "quest-high");
    // An unrelated settings change does not undo the override either.
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .display
        .centers_radius = 3.0;
    t.frames(2);
    assert_eq!(running(&t.app), "quest-high");
}
