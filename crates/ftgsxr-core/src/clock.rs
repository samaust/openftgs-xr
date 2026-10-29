// SPDX-License-Identifier: Apache-2.0

//! The playback clock T and each instance's model time τ_m (E9, §1.6, D-32).

use crate::types::{CoreError, Model, TimeSpan};

/// How the clock behaves at the ends of its active interval (E9, §1.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopMode {
    /// Wraps to the other end (the default, §1.6).
    Loop,
    /// Reflects, reversing the direction.
    PingPong,
    /// Pauses at the end.
    Once,
}

/// What an instance shows outside its span (E9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndRule {
    /// Clamps τ_m to [τ₀, τ₁] (the default).
    Hold,
    /// Leaves τ_m outside the span and flags the instance.
    Extrapolate,
}

/// Whether the clock advances (§1.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayState {
    /// T advances by speed·Δt.
    Playing,
    /// T stays.
    Paused,
}

/// The playback clock T (E9): pure, advanced by the caller's Δt.
#[derive(Clone, Debug, PartialEq)]
pub struct Clock {
    t: f64,
    speed: f64,
    state: PlayState,
    length: f64,
    mode: LoopMode,
    range: Option<(f64, f64)>,
}

impl Clock {
    /// A clock of length T_len = `length_s` seconds (E9).
    ///
    /// It starts at T = 0, speed 1, `Loop`, `Paused`, without a range (Q-T10.1).
    pub fn new(length_s: f64) -> Clock {
        // Q-T10.1: the initial state.
        Clock {
            t: 0.0,
            speed: 1.0,
            state: PlayState::Paused,
            length: valid_length(length_s),
            mode: LoopMode::Loop,
            range: None,
        }
    }

    /// Advances T by speed·`dt_s` while playing (E9).
    ///
    /// Nothing happens while paused or when T_len = 0 (E9's idle clock). The loop mode then
    /// maps T back into the active interval in closed form, whatever `dt_s` (Q-T10.1).
    pub fn advance(&mut self, dt_s: f64) {
        if self.state == PlayState::Paused || self.length <= 0.0 || !dt_s.is_finite() {
            return;
        }
        let (a, b) = self.active_interval();
        // Q-T10.1: a T outside the active interval is clamped into it first.
        let t = self.t.clamp(a, b) + self.speed * dt_s;
        let width = b - a;
        self.t = match self.mode {
            // Q-T10.1: the half-open [T_a, T_b); T_b exactly gives T_a.
            LoopMode::Loop if width > 0.0 => a + (t - a).rem_euclid(width),
            // Q-T10.1: a triangle wave of period 2·width; an odd number of reflections
            // negates the speed, so `speed()` stays dT/dt.
            LoopMode::PingPong if width > 0.0 => {
                let reflections = (t - a).div_euclid(width);
                let within = (t - a).rem_euclid(width);
                if reflections.rem_euclid(2.0) == 1.0 {
                    self.speed = -self.speed;
                    b - within
                } else {
                    a + within
                }
            }
            // Q-T10.1: once stops at T_len going forward, 0 going backwards.
            LoopMode::Once if t >= b => {
                self.state = PlayState::Paused;
                b
            }
            LoopMode::Once if t <= a => {
                self.state = PlayState::Paused;
                a
            }
            LoopMode::Once => t,
            // A degenerate range holds T at T_a.
            LoopMode::Loop | LoopMode::PingPong => a,
        };
    }

    /// Sets T (a scrub, E9).
    ///
    /// T is clamped to [0, T_len]; a non-finite value is ignored (Q-T10.1).
    pub fn seek(&mut self, t_s: f64) {
        if t_s.is_finite() {
            self.t = t_s.clamp(0.0, self.length);
        }
    }

    /// Plays or pauses (§1.6).
    pub fn set_state(&mut self, s: PlayState) {
        self.state = s;
    }

    /// Sets the speed, in [−4, 4] \ {0}, else `Err(CoreError::Value)` (E9).
    pub fn set_speed(&mut self, speed: f64) -> Result<(), CoreError> {
        if speed == 0.0 || !(-MAX_SPEED..=MAX_SPEED).contains(&speed) {
            return Err(CoreError::Value(format!("speed {speed}")));
        }
        self.speed = speed;
        Ok(())
    }

    /// Sets the loop mode (§1.6).
    pub fn set_loop(&mut self, m: LoopMode) {
        self.mode = m;
    }

    /// Sets or clears the playback range [T_a, T_b] (§1.6).
    ///
    /// The pair is ordered and clamped to [0, T_len]; a non-finite bound clears the range
    /// (Q-T10.1).
    pub fn set_range(&mut self, range: Option<(f64, f64)>) {
        self.range = range
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| {
                let clamp = |x: f64| x.clamp(0.0, self.length);
                (clamp(a.min(b)), clamp(a.max(b)))
            });
    }

    /// Sets T_len (E9).
    ///
    /// A negative or non-finite length is 0; T and the range are clamped to the new length
    /// (Q-T10.1).
    pub fn set_length(&mut self, length_s: f64) {
        self.length = valid_length(length_s);
        self.t = self.t.clamp(0.0, self.length);
        self.set_range(self.range);
    }

    /// The interval the loop mode acts within: the range in loop and ping-pong, [0, T_len]
    /// in once (§1.6, Q-T10.1).
    fn active_interval(&self) -> (f64, f64) {
        match (self.mode, self.range) {
            (LoopMode::Loop | LoopMode::PingPong, Some(range)) => range,
            _ => (0.0, self.length),
        }
    }

    /// T, seconds.
    pub fn time(&self) -> f64 {
        self.t
    }

    /// The signed speed.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Playing or paused.
    pub fn state(&self) -> PlayState {
        self.state
    }

    /// T_len, seconds.
    pub fn length(&self) -> f64 {
        self.length
    }
}

/// The largest speed magnitude (E9).
const MAX_SPEED: f64 = 4.0;

/// A length as the clock keeps it: a negative or non-finite one is 0 (Q-T10.1).
fn valid_length(length_s: f64) -> f64 {
    if length_s.is_finite() && length_s > 0.0 {
        length_s
    } else {
        0.0
    }
}

/// One instance's time mapping (E9).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelTime {
    /// The model's τ₀, τ₁ and unit.
    pub span: TimeSpan,
    /// o_m, seconds.
    pub offset_s: f64,
    /// r_m, units of τ per second.
    pub rate: f64,
    /// Hold or extrapolate outside the span.
    pub end_rule: EndRule,
}

impl ModelTime {
    /// The default mapping of a model (E9).
    pub fn from_span(span: &TimeSpan) -> ModelTime {
        let _ = span;
        todo!()
    }

    /// (τ_m, extrapolating) at clock time `t_s` (E9).
    pub fn tau_at(&self, t_s: f64) -> (f64, bool) {
        let _ = t_s;
        todo!()
    }

    /// o_m + T_τ / r_m, seconds (E9).
    pub fn length_s(&self) -> f64 {
        todo!()
    }

    /// §1.6's Δ_frame, seconds (D-32).
    pub fn frame_step_s(&self, model: &Model) -> f64 {
        let _ = model;
        todo!()
    }
}
