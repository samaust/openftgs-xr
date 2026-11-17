// SPDX-License-Identifier: Apache-2.0

//! Dv-2's refresh-rate judgement (§3.4 Dv-2, §1.12): whether the frames the `dv2` device
//! test counted over its window came at the default 72 Hz, and the first second after
//! both eyes draw, counted and reported, never judged (Q-464, #854). Built on the host
//! too, so its tests run there (#269).

use std::time::{Duration, Instant};

/// Dv-2's warm-up: its window opens this long after both eyes draw, past start-up's
/// first second (owner, 2026-11-16, #810; Q-464).
// Q-464
const WARM_UP: Duration = Duration::from_secs(2);

/// How long after both eyes draw Dv-2's window opens: the warm-up, or a level cycle's
/// length when one runs first (Q-464).
pub(crate) fn window_lead(cycle: Option<Duration>) -> Duration {
    cycle.unwrap_or(WARM_UP)
}

/// The default refresh rate (§1.12).
const RATE_HZ: f64 = 72.0;
/// The shares of 72 Hz's frames the window must hold at least and at most: a dropped
/// frame passes, a halved rate or another of the runtime's rates fails.
// Q-138
const MIN_RATE_SHARE: f64 = 0.95;
// Q-138
const MAX_RATE_SHARE: f64 = 1.05;

/// `Ok` when `frames` counted over `elapsed_s` seconds came at the default rate, else
/// which bound they missed, with the bound and the measured rate.
pub(crate) fn at_default_rate(frames: u32, elapsed_s: f64) -> Result<(), String> {
    let n = f64::from(frames);
    let expected = RATE_HZ * elapsed_s;
    let (min, max) = (MIN_RATE_SHARE * expected, MAX_RATE_SHARE * expected);
    let rate = n / elapsed_s;
    if n < min {
        Err(format!(
            "{rate:.1} Hz is below {:.0} % of {RATE_HZ} Hz: {frames} frames < {min:.0}",
            MIN_RATE_SHARE * 100.0
        ))
    } else if n > max {
        Err(format!(
            "{rate:.1} Hz is above {:.0} % of {RATE_HZ} Hz: {frames} frames > {max:.0}",
            MAX_RATE_SHARE * 100.0
        ))
    } else {
        Ok(())
    }
}

/// Start-up's first second after both eyes draw (Q-464, #854): its frames, counted and
/// logged once, never judged. With the frame times between them, for #851's measurement.
// Q-464
pub(crate) struct FirstSecond {
    start: Instant,
    last: Instant,
    steps: Vec<Duration>,
    done: bool,
}

impl FirstSecond {
    /// The second that begins with the frame at `start`, the first frame counted.
    pub(crate) fn new(start: Instant) -> FirstSecond {
        FirstSecond {
            start,
            last: start,
            // A second at 72 Hz holds 71 steps; room for a few more.
            steps: Vec::with_capacity(80),
            done: false,
        }
    }

    /// Counts the frame at `now`: `None` while the second runs; once, at the first frame
    /// after it, the lines to log: the frame count, then the times between the frames.
    pub(crate) fn frame(&mut self, now: Instant) -> Option<[String; 2]> {
        if self.done {
            return None;
        }
        if now.duration_since(self.start) < Duration::from_secs(1) {
            if now > self.last {
                self.steps.push(now - self.last);
                self.last = now;
            }
            return None;
        }
        self.done = true;
        let ms: Vec<String> = self
            .steps
            .iter()
            .map(|d| format!("{:.1}", d.as_secs_f64() * 1e3))
            .collect();
        let longest = self.steps.iter().max().copied().unwrap_or_default();
        Some([
            format!(
                "dv2: first second after both eyes draw: {} frames",
                self.steps.len() + 1
            ),
            format!(
                "dv2: first second's frame times (ms, longest {:.1}): {}",
                longest.as_secs_f64() * 1e3,
                ms.join(" ")
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_lead_is_a_2_s_warm_up_or_the_cycle() {
        assert_eq!(window_lead(None), Duration::from_secs(2));
        assert_eq!(
            window_lead(Some(Duration::from_secs(10))),
            Duration::from_secs(10)
        );
    }

    // Q-138: the band is 95 % to 105 % of 72 Hz × the elapsed time.

    #[test]
    fn at_default_rate_passes_a_session_at_72_hz() {
        assert!(at_default_rate(720, 10.0).is_ok());
        // The device run D-30 records (71.4 Hz): 715 frames in 10.01 s.
        assert!(at_default_rate(715, 10.01).is_ok());
        assert!(at_default_rate(685, 10.0).is_ok());
        assert!(at_default_rate(755, 10.0).is_ok());
    }

    #[test]
    fn at_default_rate_fails_a_halved_rate_on_the_lower_bound() {
        let e = at_default_rate(360, 10.0).unwrap_err();
        assert!(e.contains("below"), "{e}");
        assert!(at_default_rate(683, 10.0).is_err());
    }

    #[test]
    fn at_default_rate_fails_any_other_quest_rate_on_the_upper_bound() {
        // §1.12: the Quest 3's other rates, 80 to 120 Hz.
        for hz in [80, 90, 96, 100, 120] {
            let e = at_default_rate(hz * 10, 10.0).unwrap_err();
            assert!(e.contains("above"), "{hz} Hz: {e}");
        }
        assert!(at_default_rate(757, 10.0).is_err());
    }

    #[test]
    fn first_second_counts_the_frames_of_its_second_once() {
        let t0 = Instant::now();
        let step = Duration::from_micros(13_889);
        let mut s = FirstSecond::new(t0);
        // 72 frames in [0, 1 s): the first at t0, 71 more.
        for i in 1..72 {
            assert!(s.frame(t0 + step * i).is_none());
        }
        let [count, times] = s.frame(t0 + Duration::from_secs(1)).expect("the lines");
        assert_eq!(count, "dv2: first second after both eyes draw: 72 frames");
        let n: u32 = count
            .strip_prefix("dv2: first second after both eyes draw: ")
            .and_then(|r| r.strip_suffix(" frames"))
            .and_then(|n| n.parse().ok())
            .expect("an integer");
        assert_eq!(n, 72);
        assert!(times.starts_with("dv2: first second's frame times (ms, longest 13.9): 13.9 "));
        assert_eq!(times.rsplit_once("): ").unwrap().1.split(' ').count(), 71);
        assert!(
            s.frame(t0 + Duration::from_secs(2)).is_none(),
            "logged once"
        );
    }

    #[test]
    fn first_second_reports_a_slow_start_with_its_longest_frame() {
        let t0 = Instant::now();
        let mut s = FirstSecond::new(t0);
        assert!(s.frame(t0 + Duration::from_millis(500)).is_none());
        assert!(s.frame(t0 + Duration::from_millis(600)).is_none());
        let [count, times] = s.frame(t0 + Duration::from_millis(1200)).unwrap();
        assert!(count.ends_with(": 3 frames"), "{count}");
        assert!(times.contains("longest 500.0): 500.0 100.0"), "{times}");
    }
}
