// SPDX-License-Identifier: Apache-2.0

//! U-14, the `Clock` part (E9, §1.6, `cpu` lane): a clock of length 2 s at speed 2 wraps in
//! loop, reflects in ping-pong and pauses at the end in once; a playback range [0.5, 1.5];
//! speed 0 refused; speed −1 runs backwards and wraps to the end. Tolerance 10⁻¹² (exact
//! where the row says so). The behaviours E9 leaves open follow Q-19.

use approx::assert_abs_diff_eq;
use ftgsxr_core::{Clock, LoopMode, PlayState};

const TOL: f64 = 1e-12;

/// A playing clock of length 2 s at speed 2 in the given loop mode.
fn playing(mode: LoopMode) -> Clock {
    let mut clock = Clock::new(2.0);
    clock.set_speed(2.0).expect("2 is a valid speed");
    clock.set_loop(mode);
    clock.set_state(PlayState::Playing);
    clock
}

#[test]
fn u14_loop_wraps_at_the_end() {
    let mut clock = playing(LoopMode::Loop);
    clock.advance(0.75); // T = 1.5
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    clock.advance(0.5); // 1.5 + 1.0 = 2.5 → 0.5
    assert_abs_diff_eq!(clock.time(), 0.5, epsilon = TOL);
    assert_eq!(clock.state(), PlayState::Playing);
    assert_eq!(clock.speed(), 2.0);
}

#[test]
fn u14_loop_reaching_the_end_exactly_gives_the_start() {
    // Q-19: loop acts on the half-open [0, T_len).
    let mut clock = playing(LoopMode::Loop);
    clock.advance(1.0);
    assert_eq!(clock.time(), 0.0);
}

#[test]
fn u14_ping_pong_reflects_and_runs_back() {
    let mut clock = playing(LoopMode::PingPong);
    clock.advance(0.75); // T = 1.5
    clock.advance(0.5); // 1.5 + 1.0 = 2.5 reflects to 1.5
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    // Q-19: the speed is the signed dT/dt, negated by the reflection.
    assert_eq!(clock.speed(), -2.0);
    clock.advance(0.25); // 1.5 − 0.5
    assert_abs_diff_eq!(clock.time(), 1.0, epsilon = TOL);
    clock.advance(0.75); // 1.0 − 1.5 = −0.5 reflects to 0.5, forward again
    assert_abs_diff_eq!(clock.time(), 0.5, epsilon = TOL);
    assert_eq!(clock.speed(), 2.0);
    assert_eq!(clock.state(), PlayState::Playing);
}

#[test]
fn u14_ping_pong_folds_a_long_step_in_one_advance() {
    // 0 + 2·2.75 = 5.5: up to 2, down to 0, up to 1.5 (two reflections, speed kept).
    let mut clock = playing(LoopMode::PingPong);
    clock.advance(2.75);
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    assert_eq!(clock.speed(), 2.0);
    // 1.5 + 2·0.5 = 2.5: one reflection, to 1.5 going down.
    clock.advance(0.5);
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    assert_eq!(clock.speed(), -2.0);
}

#[test]
fn u14_once_pauses_at_the_end() {
    let mut clock = playing(LoopMode::Once);
    clock.advance(0.75); // T = 1.5
    assert_eq!(clock.state(), PlayState::Playing);
    clock.advance(0.5); // 2.5 stops at 2
    assert_eq!(clock.time(), 2.0);
    assert_eq!(clock.state(), PlayState::Paused);
    clock.advance(0.5); // paused: nothing moves
    assert_eq!(clock.time(), 2.0);
}

#[test]
fn u14_once_backwards_pauses_at_zero() {
    let mut clock = playing(LoopMode::Once);
    clock.seek(1.0);
    clock.set_speed(-1.0).expect("−1 is a valid speed");
    clock.advance(1.0); // reaches 0 exactly
    assert_eq!(clock.time(), 0.0);
    assert_eq!(clock.state(), PlayState::Paused);
}

#[test]
fn u14_loop_acts_within_the_range() {
    let mut clock = playing(LoopMode::Loop);
    clock.set_range(Some((0.5, 1.5)));
    clock.seek(1.0);
    clock.advance(0.5); // 1.0 + 1.0 = 2.0 wraps within [0.5, 1.5) to 1.0
    assert_abs_diff_eq!(clock.time(), 1.0, epsilon = TOL);
    clock.advance(0.3); // 1.6 → 0.6
    assert_abs_diff_eq!(clock.time(), 0.6, epsilon = TOL);
}

#[test]
fn u14_ping_pong_acts_within_the_range() {
    let mut clock = playing(LoopMode::PingPong);
    clock.set_range(Some((0.5, 1.5)));
    clock.seek(1.0);
    clock.advance(0.5); // 1.0 + 1.0 = 2.0 reflects at 1.5 to 1.0
    assert_abs_diff_eq!(clock.time(), 1.0, epsilon = TOL);
    assert_eq!(clock.speed(), -2.0);
    clock.advance(0.5); // 1.0 − 1.0 = 0.0 reflects at 0.5 to 1.0
    assert_abs_diff_eq!(clock.time(), 1.0, epsilon = TOL);
    assert_eq!(clock.speed(), 2.0);
}

#[test]
fn u14_a_time_outside_the_range_enters_it() {
    // Q-19: T is clamped into the range before the step.
    let mut clock = playing(LoopMode::Loop);
    clock.set_range(Some((0.5, 1.5)));
    clock.advance(0.1); // T = 0 → 0.5, then 0.5 + 0.2
    assert_abs_diff_eq!(clock.time(), 0.7, epsilon = TOL);
}

#[test]
fn u14_once_ignores_the_range() {
    // Q-19: §1.6 has loop and ping-pong act within the range; once plays to T_len.
    let mut clock = playing(LoopMode::Once);
    clock.set_range(Some((0.5, 1.5)));
    clock.seek(1.0);
    clock.advance(0.4); // 1.8, past T_b = 1.5
    assert_abs_diff_eq!(clock.time(), 1.8, epsilon = TOL);
    assert_eq!(clock.state(), PlayState::Playing);
}

#[test]
fn u14_range_is_ordered_and_clamped() {
    // Q-19: reversed bounds are ordered, bounds beyond [0, T_len] clamped.
    let mut clock = playing(LoopMode::Loop);
    clock.set_range(Some((3.0, 1.0))); // → [1, 2)
    clock.seek(0.0);
    clock.advance(0.25); // 0 → 1, + 0.5
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    clock.advance(0.5); // 2.5 → 1.5
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    clock.set_range(None);
    clock.advance(0.5); // 2.5 → 0.5 over [0, 2)
    assert_abs_diff_eq!(clock.time(), 0.5, epsilon = TOL);
}

#[test]
fn u14_speed_zero_refused() {
    let mut clock = Clock::new(2.0);
    let err = clock.set_speed(0.0).expect_err("speed 0 is refused");
    assert_eq!(err.code(), "E-FILE-VALUE");
    assert_eq!(clock.speed(), 1.0);
}

#[test]
fn u14_speed_bounds() {
    let mut clock = Clock::new(2.0);
    for ok in [4.0, -4.0, 0.1, -0.25] {
        clock.set_speed(ok).expect("in [−4, 4] \\ {0}");
        assert_eq!(clock.speed(), ok);
    }
    for bad in [4.5, -4.000_001, f64::NAN, f64::INFINITY, -0.0] {
        assert!(clock.set_speed(bad).is_err(), "{bad} is refused");
    }
    assert_eq!(clock.speed(), -0.25);
}

#[test]
fn u14_speed_minus_one_runs_backwards_and_wraps_to_the_end() {
    let mut clock = Clock::new(2.0);
    clock.set_speed(-1.0).expect("−1 is a valid speed");
    clock.set_state(PlayState::Playing);
    clock.advance(0.5); // 0 − 0.5 wraps to 1.5
    assert_abs_diff_eq!(clock.time(), 1.5, epsilon = TOL);
    clock.advance(0.5);
    assert_abs_diff_eq!(clock.time(), 1.0, epsilon = TOL);
}

#[test]
fn u14_paused_clock_stays() {
    let mut clock = Clock::new(2.0);
    assert_eq!(clock.state(), PlayState::Paused);
    clock.advance(0.5);
    assert_eq!(clock.time(), 0.0);
}

#[test]
fn u14_zero_length_clock_is_idle() {
    // E9: when every instance is single-timestamp, T_len = 0 and the clock is idle.
    let mut clock = Clock::new(0.0);
    clock.set_state(PlayState::Playing);
    clock.advance(1.0);
    clock.seek(3.0);
    assert_eq!(clock.time(), 0.0);
    assert_eq!(clock.length(), 0.0);
}

#[test]
fn u14_seek_clamps_to_the_length() {
    let mut clock = Clock::new(2.0);
    clock.seek(1.25);
    assert_eq!(clock.time(), 1.25);
    clock.seek(5.0);
    assert_eq!(clock.time(), 2.0);
    clock.seek(-1.0);
    assert_eq!(clock.time(), 0.0);
    clock.seek(f64::NAN); // Q-19: ignored
    assert_eq!(clock.time(), 0.0);
}

#[test]
fn u14_set_length_reclamps_time_and_range() {
    let mut clock = playing(LoopMode::Loop);
    clock.set_range(Some((0.5, 1.5)));
    clock.seek(1.8);
    clock.set_length(1.0); // T → 1, range → [0.5, 1]
    assert_eq!(clock.length(), 1.0);
    assert_eq!(clock.time(), 1.0);
    clock.advance(0.1); // 1 → clamped 1, + 0.2 = 1.2 wraps in [0.5, 1) to 0.7
    assert_abs_diff_eq!(clock.time(), 0.7, epsilon = TOL);
    clock.set_length(-3.0); // Q-19: taken as 0
    assert_eq!(clock.length(), 0.0);
    assert_eq!(clock.time(), 0.0);
}

/// A speed drawn from E9's domain [−4, 4] \ {0}.
fn speed() -> impl proptest::strategy::Strategy<Value = f64> {
    use proptest::prelude::*;
    (0.01f64..=4.0, any::<bool>()).prop_map(|(s, neg)| if neg { -s } else { s })
}

proptest::proptest! {
    /// U-14, proptest variant: in loop and ping-pong, T after any sequence of steps is
    /// E9's unwrapped T₀ + speed·ΣΔt wrapped (loop) or folded (ping-pong) into the range;
    /// in once it stays in [0, T_len] and pauses only at an end.
    #[test]
    fn u14_prop_clock_matches_the_unwrapped_time(
        length in 0.1f64..=10.0,
        speed in speed(),
        start in 0.0f64..=1.0,
        range in proptest::option::of((0.0f64..=1.0, 0.0f64..=1.0)),
        steps in proptest::collection::vec(0.0f64..=0.5, 1..16),
        mode in proptest::sample::select(vec![LoopMode::Loop, LoopMode::PingPong, LoopMode::Once]),
    ) {
        let mut clock = Clock::new(length);
        clock.set_speed(speed).expect("drawn within the domain");
        clock.set_loop(mode);
        clock.set_range(range.map(|(a, b)| (a * length, b * length)));
        clock.seek(start * length);
        clock.set_state(PlayState::Playing);
        // Hand-derived bounds of the active interval (Q-19).
        let (a, b) = match (mode, range) {
            (LoopMode::Once, _) | (_, None) => (0.0, length),
            (_, Some((x, y))) => ((x.min(y)) * length, (x.max(y)) * length),
        };
        let t0 = (start * length).clamp(a, b);
        let mut total = 0.0;
        for dt in &steps {
            clock.advance(*dt);
            total += dt;
            let t = clock.time();
            proptest::prop_assert!(t >= a - TOL && t <= b + TOL, "T = {t} outside [{a}, {b}]");
            if mode == LoopMode::Once && clock.state() == PlayState::Paused {
                proptest::prop_assert!(t == a || t == b);
                break;
            }
        }
        let x = t0 + speed * total - a;
        let w = b - a;
        let t = clock.time() - a;
        match mode {
            LoopMode::Loop if w > 0.0 => {
                let expected = x.rem_euclid(w);
                let d = (t - expected).abs();
                // Distance on the circle of circumference w.
                proptest::prop_assert!(d.min(w - d) <= TOL, "loop: {t} vs {expected}");
            }
            LoopMode::PingPong if w > 0.0 => {
                let m = x.rem_euclid(2.0 * w);
                let expected = if m <= w { m } else { 2.0 * w - m };
                proptest::prop_assert!((t - expected).abs() <= TOL, "ping-pong: {t} vs {expected}");
            }
            LoopMode::Once if clock.state() == PlayState::Playing => {
                proptest::prop_assert!((t - x).abs() <= TOL, "once: {t} vs {x}");
            }
            _ => {}
        }
    }
}
