// SPDX-License-Identifier: Apache-2.0

//! PRD #221's added `xr` test (§1.8 Controllers, Appendix D.2): on Monado, the action set
//! `player` with every Appendix D.2 action and the profiles' suggested bindings attaches
//! without an OpenXR error, and `XrHands` exists with each hand `None` or of source
//! `Controller` (Q-128). Not I-16 (M-5). Prerequisites: `xr_support`'s.

use crate::xr_support;

use std::time::{Duration, Instant};

use bevy_mod_xr::session::XrState;
use ftgsxr_render::SortMode;
use ftgsxr_xr::{HandSource, XrHands, XrPlugin};
use tracing::Level;
use xr_support::{Monado, capture_log, lines, xr_app};

#[test]
#[ignore = "xr"]
fn xr_controllers_player_set_attaches_and_every_hand_source_is_a_controller() {
    capture_log();
    let _monado = Monado::start();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |_| {},
    );
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    let running = Instant::now();
    while running.elapsed() < Duration::from_secs(1) {
        app.update();
    }

    let hands = app
        .world()
        .get_resource::<XrHands>()
        .expect("XrHands exists");
    for hand in [&hands.left, &hands.right].into_iter().flatten() {
        assert_eq!(hand.source, HandSource::Controller);
    }
    assert!(!hands.left_handed, "the default [locomotion] left_handed");

    let log = lines();
    assert!(
        log.iter().any(|l| l.message.contains("attached sessions!")),
        "the action set was attached"
    );
    let errors: Vec<_> = log
        .iter()
        .filter(|l| {
            l.level == Level::ERROR
                || (l.level == Level::WARN && l.message.contains("syncing actionsets"))
        })
        .collect();
    assert!(errors.is_empty(), "no OpenXR error: {errors:?}");
}
