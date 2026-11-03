// SPDX-License-Identifier: Apache-2.0

//! The desktop's log (§1.13 Logs, §1.11, Q-151): standard error and the player's own
//! size-rotated `logs/ftgsxr.log`, with per-frame messages limited to one per second per
//! site.
//!
//! A per-frame message is an event with a field named `per_frame` (Q-T277.3), as in
//! `bevy::log::warn!(per_frame = true, "…")`; every other event passes unlimited.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use bevy::log::tracing::callsite::Identifier;
use bevy::log::tracing::{Event, Subscriber};
use ftgsxr_io::IoError;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

/// The log file's name in the logs directory (§1.11).
const LOG_NAME: &str = "ftgsxr.log";
/// The size past which the file rotates (§1.11: 10 MB).
const ROTATE_BYTES: u64 = 10 * 1000 * 1000;
/// The files kept: `ftgsxr.log` and `.1` to `.4` (§1.11).
const KEPT: u32 = 5;
/// The per-frame messages' period per site (§1.13).
const RATE_PERIOD: Duration = Duration::from_secs(1);

/// Installs the desktop's subscriber (Q-151): `level` (`error`, `warn`, `info`, `debug`
/// or `trace`; any other name is `info`) to standard error and to `logs_dir/ftgsxr.log`,
/// rotated past 10 MB with five files kept. E-IO when the directory or the file cannot
/// be made, or a subscriber is already installed.
// Q-151
pub fn init_desktop_log(level: &str, logs_dir: &Path) -> Result<(), IoError> {
    let file = RotatingWriter::open(logs_dir, ROTATE_BYTES)?;
    tracing_subscriber::registry()
        .with(level_filter(level))
        .with(RateLimit::new(RATE_PERIOD))
        .with(tracing_subscriber::fmt::layer().with_writer(io::stderr))
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(Mutex::new(file)),
        )
        .try_init()
        .map_err(|e| IoError::Io(io::Error::other(e)))
}

/// §1.13's level names; any other is `info`.
fn level_filter(level: &str) -> LevelFilter {
    match level {
        "error" => LevelFilter::ERROR,
        "warn" => LevelFilter::WARN,
        "debug" => LevelFilter::DEBUG,
        "trace" => LevelFilter::TRACE,
        _ => LevelFilter::INFO,
    }
}

/// A log file rotated at a write boundary once it has reached `limit` bytes: `name.3`
/// becomes `name.4` (the oldest, `name.4`, is removed), …, `name` becomes `name.1`.
struct RotatingWriter {
    dir: PathBuf,
    limit: u64,
    file: File,
    len: u64,
}

impl RotatingWriter {
    /// Opens `dir/ftgsxr.log` for appending, creating the directory.
    fn open(dir: &Path, limit: u64) -> Result<Self, IoError> {
        fs::create_dir_all(dir)?;
        let file = Self::append(&dir.join(LOG_NAME))?;
        let len = file.metadata()?.len();
        Ok(Self {
            dir: dir.to_owned(),
            limit,
            file,
            len,
        })
    }

    fn append(path: &Path) -> io::Result<File> {
        OpenOptions::new().create(true).append(true).open(path)
    }

    /// The `k`-th file: `ftgsxr.log` for 0, `ftgsxr.log.k` after.
    fn nth(&self, k: u32) -> PathBuf {
        if k == 0 {
            self.dir.join(LOG_NAME)
        } else {
            self.dir.join(format!("{LOG_NAME}.{k}"))
        }
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        for k in (0..KEPT - 1).rev() {
            let from = self.nth(k);
            if from.exists() {
                fs::rename(&from, self.nth(k + 1))?;
            }
        }
        self.file = Self::append(&self.nth(0))?;
        self.len = 0;
        Ok(())
    }
}

impl Write for RotatingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.len >= self.limit {
            self.rotate()?;
        }
        let n = self.file.write(buf)?;
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

// Q-T277.3
/// The per-frame messages' limit (§1.13): one per `period` per call site.
// Q-T277.3
struct RateLimit {
    period: Duration,
    last: Mutex<HashMap<Identifier, Instant>>,
}

impl RateLimit {
    fn new(period: Duration) -> Self {
        Self {
            period,
            last: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the site's message at `now` passes: the first does, and then the first
    /// at least `period` after the last that passed.
    fn allow(&self, site: Identifier, now: Instant) -> bool {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        match last.get(&site) {
            Some(t) if now.saturating_duration_since(*t) < self.period => false,
            _ => {
                last.insert(site, now);
                true
            }
        }
    }
}

impl<S: Subscriber> Layer<S> for RateLimit {
    fn event_enabled(&self, event: &Event<'_>, _ctx: Context<'_, S>) -> bool {
        let meta = event.metadata();
        if meta.fields().field("per_frame").is_none() {
            return true;
        }
        self.allow(meta.callsite(), Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ftgsxr-app-desktop-log-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_file_rotates_past_the_limit_and_keeps_five() {
        // §1.11: `logs/ftgsxr.log` rotated at 10 MB, five files kept; a 10-byte limit
        // stands for the 10 MB.
        assert_eq!(ROTATE_BYTES, 10_000_000);
        let dir = scratch("rotate");
        let mut w = RotatingWriter::open(&dir, 10).unwrap();
        for k in 0..8u8 {
            // Each line is 10 bytes, so each write after the first starts a file.
            w.write_all(&[b'0' + k; 10]).unwrap();
        }
        w.flush().unwrap();
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "ftgsxr.log",
                "ftgsxr.log.1",
                "ftgsxr.log.2",
                "ftgsxr.log.3",
                "ftgsxr.log.4"
            ]
        );
        // The newest line is in `ftgsxr.log`, the oldest kept in `.4`.
        assert_eq!(fs::read(dir.join("ftgsxr.log")).unwrap(), [b'7'; 10]);
        assert_eq!(fs::read(dir.join("ftgsxr.log.1")).unwrap(), [b'6'; 10]);
        assert_eq!(fs::read(dir.join("ftgsxr.log.4")).unwrap(), [b'3'; 10]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reopened_file_keeps_its_size() {
        let dir = scratch("reopen");
        RotatingWriter::open(&dir, 10)
            .unwrap()
            .write_all(b"0123456789")
            .unwrap();
        let mut w = RotatingWriter::open(&dir, 10).unwrap();
        w.write_all(b"abc").unwrap();
        assert_eq!(fs::read(dir.join("ftgsxr.log")).unwrap(), b"abc");
        assert_eq!(fs::read(dir.join("ftgsxr.log.1")).unwrap(), b"0123456789");
        fs::remove_dir_all(&dir).unwrap();
    }

    // Not zero-sized, so the two statics have distinct addresses (`Identifier`'s identity).
    struct Site(#[allow(dead_code)] u8);
    static SITE_A: Site = Site(0);
    static SITE_B: Site = Site(1);
    impl bevy::log::tracing::callsite::Callsite for Site {
        fn set_interest(&self, _: bevy::log::tracing::subscriber::Interest) {}
        fn metadata(&self) -> &bevy::log::tracing::Metadata<'_> {
            unreachable!("the limiter reads only the identifier")
        }
    }

    #[test]
    fn per_frame_messages_pass_once_a_second_per_site() {
        // §1.13: one per second per site.
        let limit = RateLimit::new(RATE_PERIOD);
        let (a, b) = (Identifier(&SITE_A), Identifier(&SITE_B));
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        assert!(limit.allow(a.clone(), at(0)));
        assert!(!limit.allow(a.clone(), at(16)));
        assert!(
            limit.allow(b.clone(), at(16)),
            "another site has its own second"
        );
        assert!(!limit.allow(a.clone(), at(999)));
        assert!(limit.allow(a.clone(), at(1000)));
        assert!(!limit.allow(a.clone(), at(1500)));
        assert!(limit.allow(a, at(2000)));
        assert!(!limit.allow(b, at(1000)));
    }

    #[test]
    fn level_names() {
        assert_eq!(level_filter("warn"), LevelFilter::WARN);
        assert_eq!(level_filter("trace"), LevelFilter::TRACE);
        assert_eq!(level_filter("loud"), LevelFilter::INFO);
    }
}
