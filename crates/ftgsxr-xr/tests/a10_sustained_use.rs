// SPDX-License-Identifier: Apache-2.0

//! §3.6 A-10, sustained use (Q-411): twenty minutes in `quest-balanced` without a thermal
//! warning or a second with fewer than 71 frames presented (72 Hz less one, for a frame at a
//! second's edge), read from A-10's record in logcat
//! (`ftgsxr-xr`'s `display`, `docs/modules/ftgsxr-xr/display.md`).
//!
//! The verdict is a pure function of the record's lines, with `cpu` tests. The run is
//! `a10_twenty_minutes_in_quest_balanced`, outside every lane (`#[ignore = "acceptance"]`,
//! run by name), since it needs a person wearing the headset.
// Q-411, Q-442

use std::collections::BTreeMap;

/// The seconds A-10 judges: twenty minutes.
const SECONDS: u64 = 20 * 60;
/// The fewest frames a judged second may present (Q-411: "no second below 71 frames").
const MIN_FRAMES: u32 = 71;
/// The tier A-10 runs in.
const TIER: &str = "quest-balanced";

/// One `perf record:` line: a second of the display's clock.
#[derive(Clone, Debug, PartialEq)]
struct Second {
    index: u64,
    frames: u32,
    worst_ms: f64,
    tier: String,
    loaded: u64,
    dropped: Option<u64>,
}

/// Reads a `perf record:` line, wherever it sits in a logcat line.
fn parse_second(line: &str) -> Option<Second> {
    let rest = line.split_once("perf record: ")?.1;
    let words: Vec<&str> = rest.split_whitespace().collect();
    let after = |key: &str| {
        words
            .iter()
            .position(|w| *w == key)
            .and_then(|i| words.get(i + 1).copied())
    };
    Some(Second {
        index: after("second")?.parse().ok()?,
        frames: after("frames")?.parse().ok()?,
        worst_ms: after("worst")?.parse().ok()?,
        tier: after("tier")?.to_owned(),
        loaded: after("loaded")?.parse().ok()?,
        dropped: after("dropped").and_then(|d| d.parse().ok()),
    })
}

/// What a `perf settings:` line says.
#[derive(Debug, PartialEq)]
enum Settings {
    /// Whether XR\_EXT\_performance\_settings is enabled for the session.
    Extension(bool),
    /// A notification: domain, sub-domain, from level, to level.
    Notification(String, String, String, String),
}

fn parse_settings(line: &str) -> Option<Settings> {
    let rest = line.split_once("perf settings: ")?.1.trim();
    if let Some(state) = rest.strip_prefix("XR_EXT_performance_settings ") {
        return Some(Settings::Extension(state.trim() == "enabled"));
    }
    let w: Vec<&str> = rest.split_whitespace().collect();
    match w.as_slice() {
        [domain, sub, from, "->", to] => Some(Settings::Notification(
            (*domain).to_owned(),
            (*sub).to_owned(),
            (*from).to_owned(),
            (*to).to_owned(),
        )),
        _ => None,
    }
}

/// A notification level's order (the extension's NORMAL 0, WARNING 25, IMPAIRED 75);
/// an unknown level is taken as at or above warning.
fn at_or_above_warning(level: &str) -> bool {
    level != "normal"
}

/// A-10's verdict over the record's lines, fed one at a time as logcat gives them.
///
/// The twenty minutes start two seconds after the first second whose line reports a model
/// loaded: that line is written at the first frame of the next second, and the model is
/// counted once its CPU load is done, before its upload, so the load can reach into the
/// following second (Q-442). Every second of the window needs
/// its line (a second without one presented no frame) with at least 71 frames, in
/// `quest-balanced`; no thermal notification at or above warning may arrive before the
/// window ends; and the session must report XR\_EXT\_performance\_settings enabled, since
/// without it the thermal clause cannot be observed (Q-442).
#[derive(Debug, Default)]
struct Verdict {
    extension: Option<bool>,
    start: Option<u64>,
    seconds: BTreeMap<u64, Second>,
    thermal: Vec<String>,
    notifications: Vec<String>,
}

impl Verdict {
    /// Takes one logcat line.
    fn push(&mut self, line: &str) {
        if self.done() {
            return;
        }
        if let Some(s) = parse_second(line) {
            if self.start.is_none() && s.loaded > 0 {
                self.start = Some(s.index + 2);
            }
            if self.start.is_some_and(|start| s.index >= start) {
                self.seconds.insert(s.index, s);
            }
        } else if let Some(s) = parse_settings(line) {
            match s {
                Settings::Extension(on) => {
                    // A later session's report replaces an earlier one only to say absent.
                    self.extension = Some(self.extension.unwrap_or(true) && on);
                }
                Settings::Notification(domain, sub, from, to) => {
                    let text = format!("{domain} {sub} {from} -> {to}");
                    if sub == "thermal" && at_or_above_warning(&to) {
                        self.thermal.push(text.clone());
                    }
                    self.notifications.push(text);
                }
            }
        }
    }

    /// Whether a line of the window's last second, or of a later one, has arrived.
    fn done(&self) -> bool {
        match (self.start, self.seconds.keys().next_back()) {
            (Some(start), Some(last)) => *last >= start + SECONDS - 1,
            _ => false,
        }
    }

    /// The verdict: `Ok(summary)` or `Err(summary)`, the summary naming every failure.
    fn verdict(&self) -> Result<String, String> {
        let mut failures = Vec::new();
        match self.extension {
            Some(true) => {}
            Some(false) => failures.push(
                "XR_EXT_performance_settings absent: the thermal clause is unobservable".to_owned(),
            ),
            None => failures.push("no perf settings line: the record is missing".to_owned()),
        }
        let Some(start) = self.start else {
            failures.push("no record line with a model loaded".to_owned());
            return Err(failures.join("; "));
        };
        let window = start..start + SECONDS;
        let missing = window
            .clone()
            .filter(|i| !self.seconds.contains_key(i))
            .count();
        let judged: Vec<&Second> = self.seconds.range(window).map(|(_, s)| s).collect();
        let slow: Vec<&Second> = judged
            .iter()
            .copied()
            .filter(|s| s.frames < MIN_FRAMES)
            .collect();
        let other_tier: Vec<&Second> = judged.iter().copied().filter(|s| s.tier != TIER).collect();
        if missing > 0 {
            failures.push(format!(
                "{missing} of {SECONDS} seconds without a record line (no frame presented, or the run ended)"
            ));
        }
        if let Some(first) = slow.first() {
            failures.push(format!(
                "{} seconds below {MIN_FRAMES} frames, the first second {} with {}",
                slow.len(),
                first.index - start,
                first.frames
            ));
        }
        if let Some(first) = other_tier.first() {
            failures.push(format!(
                "{} seconds not in {TIER} (tier {})",
                other_tier.len(),
                first.tier
            ));
        }
        if !self.thermal.is_empty() {
            failures.push(format!(
                "thermal notification at or above warning: {}",
                self.thermal.join(", ")
            ));
        }
        let frames: Vec<u32> = judged.iter().map(|s| s.frames).collect();
        let worst = judged.iter().map(|s| s.worst_ms).fold(0.0_f64, f64::max);
        let dropped = match (
            judged.first().and_then(|s| s.dropped),
            judged.last().and_then(|s| s.dropped),
        ) {
            (Some(a), Some(b)) => format!("{}", b.saturating_sub(a)),
            _ => "-".to_owned(),
        };
        let loaded = judged.iter().map(|s| s.loaded).max().unwrap_or(0);
        let summary = format!(
            "{} of {SECONDS} seconds recorded; loaded {loaded} primitives; frames per second \
             min {} median {} max {}; worst frame interval {worst:.1} ms; runtime dropped \
             frames {dropped}; notifications [{}]",
            judged.len(),
            frames.iter().min().map_or("-".to_owned(), u32::to_string),
            median(&frames).map_or("-".to_owned(), |m| m.to_string()),
            frames.iter().max().map_or("-".to_owned(), u32::to_string),
            self.notifications.join(", ")
        );
        if failures.is_empty() {
            Ok(summary)
        } else {
            Err(format!("{}; {summary}", failures.join("; ")))
        }
    }
}

/// The upper median.
fn median(v: &[u32]) -> Option<u32> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.get(s.len() / 2).copied()
}

#[cfg(test)]
mod verdict {
    use super::*;

    /// A record line as the app logs it (`display`'s record, Q-411).
    fn second(index: u64, frames: u32, tier: &str, loaded: u64) -> String {
        format!(
            "11-12 17:00:00.000  4242  4300 I ftgsxr  : perf record: second {index} frames {frames} \
             worst 13.9 ms tier {tier} scale 1 sh 2 sort SharedCenter foveation Medium \
             loaded {loaded} dropped {}",
            index * 2
        )
    }

    const ENABLED: &str = "11-12 17:00:00.000  4242  4300 I ftgsxr  : perf settings: XR_EXT_performance_settings enabled";

    /// The session's start, three seconds without a model, the load's two slow seconds,
    /// then `seconds` seconds of `frames` each.
    fn run(seconds: u64, frames: impl Fn(u64) -> u32) -> Verdict {
        let mut v = Verdict::default();
        v.push(ENABLED);
        for i in 0..3 {
            v.push(&second(i, 40, TIER, 0));
        }
        v.push(&second(3, 20, TIER, 1_000_000));
        v.push(&second(4, 30, TIER, 1_000_000));
        for i in 5..5 + seconds {
            v.push(&second(i, frames(i), TIER, 1_000_000));
        }
        v
    }

    /// A-10: a clean twenty minutes passes, the load's two slow seconds and the seconds
    /// before them left out of the window.
    #[test]
    fn a10_verdict_passes_a_clean_twenty_minutes() {
        let v = run(SECONDS, |_| 72);
        assert!(v.done());
        let summary = v.verdict().expect("a clean run passes");
        assert!(summary.contains("1200 of 1200 seconds"), "{summary}");
        assert!(summary.contains("min 72"), "{summary}");
        assert!(summary.contains("loaded 1000000 primitives"), "{summary}");
    }

    fn notification(text: &str) -> String {
        format!("11-12 17:05:00.000  4242  4300 W ftgsxr  : perf settings: {text}")
    }

    /// A-10: a thermal notification at or above warning fails, whichever level it comes
    /// from; a rendering warning and a thermal return to normal do not.
    #[test]
    fn a10_verdict_fails_on_a_thermal_warning() {
        for text in [
            "gpu thermal normal -> warning",
            "cpu thermal warning -> impaired",
            "cpu thermal impaired -> warning",
        ] {
            let mut v = run(600, |_| 72);
            v.push(&notification(text));
            for i in 605..5 + SECONDS {
                v.push(&second(i, 72, TIER, 1_000_000));
            }
            let e = v.verdict().expect_err(text);
            assert!(e.contains("thermal notification"), "{e}");
        }
        let mut v = run(600, |_| 72);
        v.push(&notification("gpu rendering normal -> warning"));
        v.push(&notification("gpu thermal warning -> normal"));
        for i in 605..5 + SECONDS {
            v.push(&second(i, 72, TIER, 1_000_000));
        }
        let ok = v.verdict().expect("no thermal warning");
        assert!(ok.contains("gpu rendering normal -> warning"), "{ok}");
    }

    /// A-10: one second below 71 frames fails, and so does a second without a line;
    /// 71 frames passes.
    #[test]
    fn a10_verdict_fails_on_a_second_below_71_frames() {
        let e = run(SECONDS, |i| if i == 700 { 70 } else { 72 })
            .verdict()
            .expect_err("70 frames");
        assert!(
            e.contains("1 seconds below 71 frames, the first second 695 with 70"),
            "{e}"
        );
        assert!(run(SECONDS, |_| 71).verdict().is_ok());
        let mut v = run(100, |_| 72);
        for i in 106..5 + SECONDS + 1 {
            v.push(&second(i, 72, TIER, 1_000_000));
        }
        let e = v.verdict().expect_err("a missing second");
        assert!(e.contains("1 of 1200 seconds without a record line"), "{e}");
    }

    /// The window stays open until its last second: a short run fails, as does a run
    /// with no model loaded, without the extension, or in another tier.
    #[test]
    fn verdict_needs_the_whole_window_the_extension_and_the_tier() {
        let short = run(SECONDS - 1, |_| 72);
        assert!(!short.done());
        assert!(
            short
                .verdict()
                .expect_err("short")
                .contains("1 of 1200 seconds")
        );
        let mut none = Verdict::default();
        none.push(ENABLED);
        none.push(&second(0, 72, TIER, 0));
        assert!(
            none.verdict()
                .expect_err("no model")
                .contains("no record line with a model")
        );
        let mut absent = Verdict::default();
        absent.push("I ftgsxr  : perf settings: XR_EXT_performance_settings absent");
        absent.push(&second(0, 72, TIER, 1));
        for i in 1..=SECONDS {
            absent.push(&second(i, 72, TIER, 1));
        }
        assert!(
            absent
                .verdict()
                .expect_err("absent")
                .contains("unobservable")
        );
        let mut other = Verdict::default();
        other.push(ENABLED);
        for i in 0..=SECONDS {
            other.push(&second(i, 72, "quest-low", 1));
        }
        assert!(
            other
                .verdict()
                .expect_err("tier")
                .contains("not in quest-balanced")
        );
    }
}

/// The budget model's name (Q-410), pushed into the app's models directory, which the
/// Library tab lists.
const BUDGET_MODEL: &str = "kitchen_1m.safetensors";
/// The app's models directory on the device (`AppPaths.models`, Q-15).
const MODELS_DIR: &str = "/sdcard/Android/data/io.github.samaust.ftgsxr/files/models";
/// How long the owner has to load the budget model after the launch.
const LOAD_WITHIN: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// §3.6 A-10, by the owner (Q-411): installs `FTGSXR_QUEST_APK`, puts the budget model in
/// the app's models directory unless it is there, launches the app as the owner would
/// (no extra, the settings as they are), follows logcat under `ftgsxr` and gives the
/// verdict once twenty minutes after the model's load are recorded. The owner puts on
/// the headset, opens `kitchen_1m.safetensors` from the Library tab in `quest-balanced`
/// and uses the player. Prerequisites: `dv_support`'s and
/// `$OPENFTGS_DATASETS/xr-fixtures/kitchen_1m.safetensors`; `[general] log_level` at
/// info or below. Run with `--ignored --exact --nocapture`.
#[test]
#[ignore = "acceptance"]
fn a10_twenty_minutes_in_quest_balanced() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use crate::dv_support::{ACTIVITY, TAG, adb, adb_path, apk, device, install, running, stop};

    let model = std::path::PathBuf::from(
        std::env::var_os("OPENFTGS_DATASETS")
            .expect("$OPENFTGS_DATASETS names the datasets (the budget model, Q-410)"),
    )
    .join("xr-fixtures")
    .join(BUDGET_MODEL);
    let size = std::fs::metadata(&model)
        .unwrap_or_else(|e| panic!("the budget model at {}: {e}", model.display()))
        .len();
    let apk = apk();
    device();
    install(&apk);
    stop();
    let on_device = format!("{MODELS_DIR}/{BUDGET_MODEL}");
    let there = adb(&[
        "shell",
        &format!("stat -c %s {on_device} 2>/dev/null || true"),
    ]);
    if there.trim() != size.to_string() {
        adb(&["shell", "mkdir", "-p", MODELS_DIR]);
        adb(&["push", &model.to_string_lossy(), &on_device]);
    }
    adb(&["logcat", "-c"]);
    let mut logcat = Command::new(adb_path())
        .args(["logcat", "-v", "brief", "-s", &format!("{TAG}:*")])
        .stdout(Stdio::piped())
        .spawn()
        .expect("adb logcat runs");
    let (tx, rx) = mpsc::channel();
    let out = logcat.stdout.take().expect("logcat's output");
    std::thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    adb(&["shell", "am", "start", "-S", "-n", ACTIVITY]);
    let launched = Instant::now();
    println!(
        "A-10: put on the headset, open {BUDGET_MODEL} from the Library tab in {TIER}, and use \
         the player; the twenty minutes start the second after it loads"
    );
    let mut v = Verdict::default();
    let mut lines = 0u64;
    let mut checked = Instant::now();
    let ended = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(line) => {
                if line.contains("perf record: ") {
                    lines += 1;
                    if lines.is_multiple_of(60) {
                        println!("{line}");
                    }
                }
                if line.contains("perf settings: ") {
                    println!("{line}");
                }
                v.push(&line);
                if v.done() {
                    break None;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break Some("adb logcat ended"),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if checked.elapsed() > Duration::from_secs(10) {
            checked = Instant::now();
            if launched.elapsed() > Duration::from_secs(20) && !running() {
                break Some("the app's process ended");
            }
            if lines == 0 && launched.elapsed() > Duration::from_secs(90) {
                break Some("no perf record line in 90 s: is [general] log_level info?");
            }
            if v.start.is_none() && launched.elapsed() > LOAD_WITHIN {
                break Some("no model loaded within 15 minutes");
            }
        }
    };
    let _ = logcat.kill();
    let _ = logcat.wait();
    stop();
    let verdict = v.verdict();
    match (&verdict, ended) {
        (Ok(summary), _) => println!("A-10 PASS {summary}"),
        (Err(summary), why) => println!(
            "A-10 FAIL {}{summary}",
            why.map_or(String::new(), |w| format!("{w}; "))
        ),
    }
    verdict.expect("A-10 passes");
}
