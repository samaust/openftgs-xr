// SPDX-License-Identifier: Apache-2.0

//! Dv-2's refresh-rate judgement (§3.4 Dv-2, §1.12): whether the frames the `dv2` device
//! test counted over its window came at the default 72 Hz. Built on the host too, so its
//! tests run there (#269).

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

#[cfg(test)]
mod tests {
    use super::*;

    // Q-138: the band is 95 % to 105 % of 72 Hz × the elapsed time.

    #[test]
    fn at_default_rate_passes_a_session_at_72_hz() {
        assert!(at_default_rate(720, 10.0).is_ok());
        // The device run Q-142 records: 715 frames in 10.01 s.
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
}
