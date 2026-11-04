// SPDX-License-Identifier: Apache-2.0

//! I-16 (`xr` lane, §3.2) as Q-271 reads it: on Monado, with the session running, a
//! synthetic `XrHands` injected into the adapter (`xr_input`'s system reads it before
//! `tracking` writes the frame's own): the controller path (`source: Controller`, the
//! trigger rising past 0.6 with an aim pose) produces `Select` and `Pointer`; the hands
//! path (`source: Hand`) produces grabs at the thresholds (Appendix D.3: ≥ 0.7 begins,
//! < 0.4 ends). Monado's simulated controllers produce no input of their own, and the
//! OpenXR binding is M-3's `xr_controllers` test's. Prerequisites: `xr_support`'s.

use std::time::{Duration, Instant};

use bevy::app::{App, First, Update};
use bevy::ecs::message::MessageReader;
use bevy::math::{Quat, Vec2, Vec3};
use bevy::prelude::{Res, ResMut, Resource};
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{Hand, Intent};
use ftgsxr_core::Pose;
use ftgsxr_render::SortMode;
use ftgsxr_xr::{HandSource, TrackedHand, XrHands, XrPlugin};
use tracing::Level;

use crate::xr_support::{Monado, capture_log, lines, xr_app};

/// The hands the next frame's adapter reads.
#[derive(Resource, Default)]
struct Script(Option<XrHands>);

/// The intents of the frames so far.
#[derive(Resource, Default)]
struct Seen(Vec<Intent>);

fn inject(script: Res<Script>, mut hands: ResMut<XrHands>) {
    if let Some(h) = script.0 {
        *hands = h;
    }
}

fn collect(mut intents: MessageReader<Intent>, mut seen: ResMut<Seen>) {
    seen.0.extend(intents.read().copied());
}

fn hand(source: HandSource) -> TrackedHand {
    TrackedHand {
        aim: Pose {
            translation: Vec3::new(0.2, 1.3, -0.1),
            rotation: Quat::from_rotation_y(0.3),
            scale: 1.0,
        },
        grip: Pose::IDENTITY,
        pinch: 0.0,
        pinch_middle: 0.0,
        squeeze: 0.0,
        trigger: 0.0,
        stick: Vec2::ZERO,
        primary: false,
        secondary: false,
        menu: false,
        source,
        tracked: true,
    }
}

/// Runs one frame with `right` as the right hand; the frame's intents.
fn frame(app: &mut App, right: TrackedHand) -> Vec<Intent> {
    app.world_mut().resource_mut::<Script>().0 = Some(XrHands {
        left: None,
        right: Some(right),
        head: Pose::IDENTITY,
        left_handed: false,
    });
    app.world_mut().resource_mut::<Seen>().0.clear();
    app.update();
    app.world().resource::<Seen>().0.clone()
}

fn running_app() -> App {
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |app| {
            app.init_resource::<Script>()
                .init_resource::<Seen>()
                .add_systems(First, inject)
                .add_systems(Update, collect);
        },
    );
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    app
}

#[test]
#[ignore = "xr"]
fn xr_i16_controller_select_and_pointer_and_hand_grabs_at_the_thresholds() {
    capture_log();
    let _monado = Monado::start();
    let mut app = running_app();

    // The controller path: the aim pose is the ray; the trigger past 0.6 selects.
    let c = hand(HandSource::Controller);
    let out = frame(&mut app, c);
    let want_dir = (c.aim.rotation * Vec3::NEG_Z).normalize();
    let pointer = out.iter().find_map(|i| match *i {
        Intent::Pointer { hand, origin, dir } => Some((hand, origin, dir)),
        _ => None,
    });
    let (h, origin, dir) = pointer.expect("a Pointer from the aim pose");
    assert_eq!(h, Hand::Right);
    assert!(origin.distance(c.aim.translation) < 1e-6);
    assert!(dir.distance(want_dir) < 1e-6);
    assert!(!out.iter().any(|i| matches!(i, Intent::Select { .. })));
    let out = frame(&mut app, TrackedHand { trigger: 0.6, ..c });
    assert!(
        !out.iter().any(|i| matches!(i, Intent::Select { .. })),
        "0.6 is not past 0.6"
    );
    let out = frame(&mut app, TrackedHand { trigger: 0.65, ..c });
    assert!(out.contains(&Intent::Select {
        hand: Hand::Right,
        pressed: true
    }));
    let out = frame(&mut app, c);
    assert!(out.contains(&Intent::Select {
        hand: Hand::Right,
        pressed: false
    }));

    // The hands path: the index pinch grabs at 0.7 and lets go below 0.4.
    let h = hand(HandSource::Hand);
    let pinch = |v: f32| TrackedHand { pinch: v, ..h };
    let begins = |out: &[Intent]| {
        out.iter()
            .filter(|i| {
                matches!(
                    i,
                    Intent::GrabBegin {
                        hand: Hand::Right,
                        ..
                    }
                )
            })
            .count()
    };
    let ends = |out: &[Intent]| {
        out.iter()
            .filter(|i| matches!(i, Intent::GrabEnd { hand: Hand::Right }))
            .count()
    };
    assert_eq!(begins(&frame(&mut app, pinch(0.69))), 0);
    assert_eq!(begins(&frame(&mut app, pinch(0.7))), 1);
    let out = frame(&mut app, pinch(0.4));
    assert_eq!((begins(&out), ends(&out)), (0, 0));
    assert!(out.iter().any(|i| matches!(
        i,
        Intent::GrabMove {
            hand: Hand::Right,
            ..
        }
    )));
    assert_eq!(ends(&frame(&mut app, pinch(0.39))), 1);

    let errors: Vec<_> = lines()
        .into_iter()
        .filter(|l| l.level == Level::ERROR)
        .collect();
    assert!(errors.is_empty(), "no error-level log: {errors:?}");
}
