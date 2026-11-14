// SPDX-License-Identifier: Apache-2.0

//! The Quest's per-frame limit (§1.13 Logs, Q-447): a `log` record carrying Q-167's
//! `per_frame` field passes once a second per site, the site being the record's source
//! file and line. This is the pure part, built on the host too so its tests run there;
//! the logger around android_logger is `android`'s.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Whether `text`, a record's text as tracing's `log` bridge writes it, carries Q-167's
/// `per_frame` field: ` per_frame=<value>` after the message, or `per_frame=<value>`
/// first when the event has no message (Q-447).
// Q-447
pub(crate) fn is_per_frame(text: &str) -> bool {
    text.starts_with("per_frame=") || text.contains(" per_frame=")
}

/// The per-frame messages' period per site (§1.13).
pub(crate) const PER_FRAME_PERIOD: Duration = Duration::from_secs(1);

/// One per `period` per site, a site being a record's source file and line (Q-447).
// Q-447
pub(crate) struct SiteLimit {
    period: Duration,
    last: Mutex<HashMap<(String, u32), Instant>>,
}

impl SiteLimit {
    /// A limit with no site seen yet.
    pub(crate) fn new(period: Duration) -> Self {
        Self {
            period,
            last: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the site's message at `now` passes: the first does, and then the first at
    /// least `period` after the last that passed.
    pub(crate) fn allow(&self, file: &str, line: u32, now: Instant) -> bool {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        match last.entry((file.to_owned(), line)) {
            Entry::Occupied(e) if now.saturating_duration_since(*e.get()) < self.period => false,
            Entry::Occupied(mut e) => {
                e.insert(now);
                true
            }
            Entry::Vacant(e) => {
                e.insert(now);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Q-447: tracing's bridge writes the message, then each other field as ` name=value`.
    #[test]
    fn is_per_frame_finds_the_field_after_the_message() {
        assert!(is_per_frame(
            "splat layer 1512x1584 for a 1680x1760 view at render scale 0.9 per_frame=true"
        ));
        assert!(!is_per_frame(
            "splat layer 1512x1584 for a 1680x1760 view at render scale 0.9"
        ));
    }

    // Q-447: an event with no message starts with its first field.
    #[test]
    fn is_per_frame_finds_the_field_first_when_there_is_no_message() {
        assert!(is_per_frame("per_frame=true"));
        assert!(is_per_frame("per_frame=true layer=3"));
        assert!(!is_per_frame("layer=3"));
    }

    // §1.13: one per second per site; the first passes, then the first a full period
    // after the last that passed.
    #[test]
    fn site_limit_passes_the_first_then_one_per_period() {
        let limit = SiteLimit::new(PER_FRAME_PERIOD);
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        assert!(limit.allow("src/render.rs", 811, at(0)));
        assert!(!limit.allow("src/render.rs", 811, at(14)));
        assert!(!limit.allow("src/render.rs", 811, at(999)));
        assert!(limit.allow("src/render.rs", 811, at(1000)));
        assert!(!limit.allow("src/render.rs", 811, at(1500)));
        assert!(limit.allow("src/render.rs", 811, at(2014)));
    }

    // Q-447: a site is a file and a line; another line or another file has its own second.
    #[test]
    fn site_limit_keeps_one_clock_per_file_and_line() {
        let limit = SiteLimit::new(PER_FRAME_PERIOD);
        let t = Instant::now();
        assert!(limit.allow("src/render.rs", 811, t));
        assert!(limit.allow("src/render.rs", 812, t));
        assert!(limit.allow("src/other.rs", 811, t));
        assert!(!limit.allow("src/render.rs", 811, t));
        assert!(!limit.allow("src/other.rs", 811, t));
    }
}
