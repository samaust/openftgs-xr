// SPDX-License-Identifier: Apache-2.0

//! The Quest's per-frame limit (§1.13 Logs, Q-167, Q-445, #793, `dv` lane): the runner's
//! `log-limit` logs, on every frame for a few seconds, a line from one site marked with
//! Q-167's `per_frame` field and a line from another, unmarked site, and reports how many
//! of each it logged. By logcat's monotonic timestamps at microsecond resolution, the
//! driver finds at least one marked line, no two marked lines less than one second
//! apart, and as many unmarked lines as the runner logged. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{TAG, adb, apk, device, install, launch, stop, verdict};

/// The main log buffer for the run, as R-17's (the runtime's lines are many).
const LOG_BUFFER: &str = "16M";
/// §1.13: one per second per site, in microseconds.
const PERIOD_US: u64 = 1_000_000;

/// A `-v monotonic,usec` line's time in microseconds since boot and its text, or `None`
/// for a line with no such time (logcat's `---------` headers).
fn timed(line: &str) -> Option<(u64, &str)> {
    let line = line.trim_start();
    let (stamp, text) = line.split_once(char::is_whitespace)?;
    let (s, us) = stamp.split_once('.')?;
    if us.len() != 6 {
        return None;
    }
    Some((
        s.parse::<u64>().ok()? * 1_000_000 + us.parse::<u64>().ok()?,
        text,
    ))
}

/// The judgement of the run's lines against the runner's count of unmarked lines;
/// `Err` names the first clause that fails, `Ok` the marked lines' count.
fn judge(lines: &[String], unmarked_logged: u32) -> Result<usize, String> {
    let mut marked = Vec::new();
    let mut unmarked = 0u32;
    for l in lines {
        let Some((t, text)) = timed(l) else { continue };
        if text.contains("log-limit: marked ") {
            marked.push((t, l));
        } else if text.contains("log-limit: unmarked ") {
            unmarked += 1;
        }
    }
    if marked.is_empty() {
        return Err("no marked line".into());
    }
    for pair in marked.windows(2) {
        let ((t0, l0), (t1, l1)) = (pair[0], pair[1]);
        if t1.saturating_sub(t0) < PERIOD_US {
            return Err(format!(
                "two marked lines {} µs apart: {l0} / {l1}",
                t1.saturating_sub(t0)
            ));
        }
    }
    if unmarked != unmarked_logged {
        return Err(format!(
            "{unmarked} unmarked lines in logcat, the runner logged {unmarked_logged}"
        ));
    }
    Ok(marked.len())
}

/// The runner's counts from its verdict's detail, `marked M unmarked U`.
fn counts(detail: &str) -> Option<(u32, u32)> {
    let words: Vec<&str> = detail.split_whitespace().collect();
    match words.as_slice() {
        ["marked", m, "unmarked", u] => Some((m.parse().ok()?, u.parse().ok()?)),
        _ => None,
    }
}

#[test]
#[ignore = "dv"]
fn dv_log_limit_marked_lines_once_a_second_unmarked_all() {
    let apk = apk();
    device();
    install(&apk);
    adb(&["logcat", "-G", LOG_BUFFER]);
    let start = launch("log-limit");
    let result = verdict("log-limit", start, Duration::from_secs(60));
    let lines: Vec<String> = adb(&[
        "logcat",
        "-d",
        "-v",
        "monotonic,usec",
        "-s",
        &format!("{TAG}:*"),
    ])
    .lines()
    .map(str::to_owned)
    .collect();
    stop();
    let (pass, detail, _) = result.expect("a log-limit verdict line in logcat");
    assert!(pass, "log-limit FAIL {detail}");
    let (marked_logged, unmarked_logged) = counts(&detail).expect("the runner's counts");
    let marked = judge(&lines, unmarked_logged).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "log-limit: {marked} of {marked_logged} marked lines and {unmarked_logged} unmarked \
         lines reached logcat"
    );
}

/// A `-v monotonic,usec` line of the app's at `t` seconds since boot.
fn line(t: &str, text: &str) -> String {
    format!("{t}  4321  4400 I ftgsxr  : {text}")
}

/// The driver's judgement (private to it, so `cpu` tests of its own, Q-445) over
/// hand-written `-v monotonic,usec` lines.
#[test]
fn judge_passes_marked_lines_a_second_apart_and_every_unmarked_line() {
    let lines = vec![
        "--------- beginning of main".to_owned(),
        line("  120.000100", "log-limit: marked 0 per_frame=true"),
        line("  120.000200", "log-limit: unmarked 0"),
        line("  120.013900", "log-limit: unmarked 1"),
        line("  121.000100", "log-limit: marked 72 per_frame=true"),
        line("  121.000200", "log-limit: unmarked 2"),
        line(
            "  121.001000",
            "ftgsxr-test: log-limit PASS marked 73 unmarked 3",
        ),
    ];
    assert_eq!(judge(&lines, 3), Ok(2));
}

#[test]
fn judge_fails_two_marked_lines_under_a_second_apart() {
    let lines = vec![
        line("120.000100", "log-limit: marked 0 per_frame=true"),
        line("121.000099", "log-limit: marked 72 per_frame=true"),
    ];
    let e = judge(&lines, 0).unwrap_err();
    assert!(e.contains("999999 µs apart"), "{e}");
}

#[test]
fn judge_fails_without_a_marked_line_or_with_a_lost_unmarked_line() {
    let lines = vec![line("120.000200", "log-limit: unmarked 0")];
    assert_eq!(judge(&lines, 1), Err("no marked line".into()));
    let lines = vec![
        line("120.000100", "log-limit: marked 0 per_frame=true"),
        line("120.000200", "log-limit: unmarked 0"),
    ];
    assert!(judge(&lines, 2).unwrap_err().contains("1 unmarked lines"));
}

#[test]
fn counts_reads_the_verdict_detail() {
    assert_eq!(counts("marked 288 unmarked 288"), Some((288, 288)));
    assert_eq!(counts("no running session"), None);
}
