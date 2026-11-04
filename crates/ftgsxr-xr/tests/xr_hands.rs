// SPDX-License-Identifier: Apache-2.0

//! `tracking`'s hands on Monado (`xr` lane, #459): with `XrPlugin.hands` set, the session
//! runs as in I-15 and the M-3 controller test, the hand trackers are created (no
//! platform: Q-287) and located each frame through the raw locate, the hand
//! extensions Monado lacks each log one E-XR-UNSUPPORTED warning (Q-134), and nothing is
//! logged at error level.

use std::time::{Duration, Instant};

use bevy_mod_xr::session::XrState;
use ftgsxr_render::SortMode;
use ftgsxr_xr::{HandSource, XrHands, XrPlugin};
use tracing::Level;

use crate::xr_support::{Monado, capture_log, lines, xr_app};

#[test]
#[ignore = "xr"]
fn xr_hands_requested_on_monado_keeps_controllers_and_logs_no_error() {
    capture_log();
    let _monado = Monado::start();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: true,
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
    while running.elapsed() < Duration::from_secs(2) {
        app.update();
    }

    let hands = app
        .world()
        .get_resource::<XrHands>()
        .expect("XrHands exists");
    for hand in [&hands.left, &hands.right].into_iter().flatten() {
        // Monado has no aim extension: a located hand is untracked (Q-285).
        assert!(hand.source == HandSource::Controller || !hand.tracked);
    }

    let log = lines();
    assert!(
        log.iter()
            .any(|l| l.message.contains("hand trackers created")),
        "the trackers were created"
    );
    for name in [
        "XR_FB_hand_tracking_aim",
        "XR_META_hand_tracking_frequency_hint",
        "XR_META_simultaneous_hands_and_controllers",
    ] {
        let n = log
            .iter()
            .filter(|l| {
                l.level == Level::WARN
                    && l.message.contains("E-XR-UNSUPPORTED")
                    && l.message.contains(name)
            })
            .count();
        assert_eq!(n, 1, "one E-XR-UNSUPPORTED warning for {name}");
    }
    let errors: Vec<_> = log.iter().filter(|l| l.level == Level::ERROR).collect();
    assert!(errors.is_empty(), "no error-level log: {errors:?}");
    let locate: Vec<_> = log
        .iter()
        .filter(|l| l.message.contains("could not be located"))
        .collect();
    assert!(locate.is_empty(), "every locate succeeds: {locate:?}");
}
