// SPDX-License-Identifier: Apache-2.0

//! R-17 on the headset (§3.2, E16, D-47, §1.12 at revision 56; #730, Q-431, `dv`
//! lane): the driver pushes P1 as `dv/probe.safetensors` (or, with `FTGSXR_R17_ROWS`
//! set, the spikes' every-row-persistent model of that many rows, Q-420), launches the
//! runner's `r17` in `quest-balanced` and judges the log: with `budget_override_ms` at
//! 0.1 the knobs step down in E16's order and counts (render scale 1.0 to 0.6 in four
//! steps, foveation medium to high, SH cap 2 to 1, no sort-mode step, since the tier
//! already sorts shared-center), each after 10 blocks over; at 1,000 they step back in
//! the reverse order, each after 120 under (exact counts). The foveation `display`
//! applies follows the knob (high, then medium), and no swapchain is re-created during
//! the run (the tier's render scale sizes the splat layer, §1.12). Prerequisites:
//! `dv_support`'s.

use crate::dv_spikes::every_row_persistent;
use crate::dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{apk, clear, device, install, launch, log, push, stop, verdict};
use ftgsxr_core::write_exchange;

/// The variable naming a row count for a heavier model than P1.
const ROWS_VAR: &str = "FTGSXR_R17_ROWS";
/// The main log buffer for the run, as the spikes' (the runtime's lines are many).
const LOG_BUFFER: &str = "16M";

/// The steps R-17 expects in `quest-balanced` (E16, §1.12's tier table): direction,
/// change, blocks since the last step.
const EXPECTED: [(&str, &str, u32); 12] = [
    ("down", "RenderScale 1.0 -> 0.9", 10),
    ("down", "RenderScale 0.9 -> 0.8", 10),
    ("down", "RenderScale 0.8 -> 0.7", 10),
    ("down", "RenderScale 0.7 -> 0.6", 10),
    ("down", "Foveation Medium -> High", 10),
    ("down", "ShCap 2 -> 1", 10),
    ("up", "ShCap 1 -> 2", 120),
    ("up", "Foveation High -> Medium", 120),
    ("up", "RenderScale 0.6 -> 0.7", 120),
    ("up", "RenderScale 0.7 -> 0.8", 120),
    ("up", "RenderScale 0.8 -> 0.9", 120),
    ("up", "RenderScale 0.9 -> 1.0", 120),
];

/// The index of the first line containing `needle`.
fn find(lines: &[String], needle: &str) -> Option<usize> {
    lines.iter().position(|l| l.contains(needle))
}

/// R-17's judgement of the run's log (lines in logcat order); `Err` names the first
/// clause that fails.
fn judge(lines: &[String]) -> Result<(), String> {
    let start = find(lines, "r17: budget 0.1 ms").ok_or("no `r17: budget 0.1 ms` line")?;
    // Up to the verdict: after it the governor runs on the display's own budget, and may
    // step a heavier model again.
    let end = lines[start..]
        .iter()
        .position(|l| l.contains("ftgsxr-test: r17 "))
        .map_or(lines.len(), |i| start + i);
    let run = &lines[start..end];
    // Each step: `r17: step <n> <down|up> <change> after <k> blocks, GPU total …`.
    let mut steps = Vec::new();
    for (i, l) in run.iter().enumerate() {
        let Some(rest) = l.split_once("r17: step ").map(|(_, r)| r) else {
            continue;
        };
        let mut words = rest.splitn(3, ' ');
        let (_, dir, tail) = (words.next(), words.next(), words.next());
        let (dir, tail) = (dir.ok_or(l.clone())?, tail.ok_or(l.clone())?);
        let (change, after) = tail.split_once(" after ").ok_or(l.clone())?;
        let blocks: u32 = after
            .split_whitespace()
            .next()
            .and_then(|w| w.parse().ok())
            .ok_or(l.clone())?;
        steps.push((i, dir.to_owned(), change.to_owned(), blocks));
    }
    let got: Vec<(&str, &str, u32)> = steps
        .iter()
        .map(|(_, d, c, b)| (d.as_str(), c.as_str(), *b))
        .collect();
    if got != EXPECTED {
        return Err(format!("the steps {got:?}, not E16's {EXPECTED:?}"));
    }
    // Every block of each way on its side of E16's threshold, so the streaks were whole.
    for (needle, side) in [
        ("r17: down held ", "not over"),
        ("r17: up held ", "not under"),
    ] {
        let l = run
            .iter()
            .find(|l| l.contains(needle))
            .ok_or(format!("no `{needle}` line"))?;
        if !l.contains(&format!(" 0 blocks {side}")) {
            return Err(format!("blocks off the way's side: {l}"));
        }
    }
    let governor = run.iter().filter(|l| l.contains("governor step")).count();
    if governor != EXPECTED.len() {
        return Err(format!("{governor} `governor step` lines, not 12"));
    }
    // The foveation `display` applied after the start: high after the step down, medium
    // after the step back, nothing else.
    let fov: Vec<(usize, &str)> = run
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let rest = l.split_once("foveation ")?.1;
            rest.contains("applied to the swapchain")
                .then(|| (i, rest.split_whitespace().next().unwrap_or("")))
        })
        .collect();
    let (down_at, up_at) = (steps[4].0, steps[7].0);
    match fov.as_slice() {
        [(h, "High"), (m, "Medium")] if *h > down_at && *h < up_at && *m > up_at => {}
        other => {
            return Err(format!(
                "foveation applied {other:?}, not High after line {down_at} then Medium after line {up_at}"
            ));
        }
    }
    // §1.12 at revision 56: the governor's render scale never reaches the swapchain.
    if let Some(l) = run.iter().find(|l| {
        l.contains("re-creating the session")
            || (l.contains("swapchain ") && l.contains(" at render scale "))
    }) {
        return Err(format!("a swapchain re-creation during the run: {l}"));
    }
    Ok(())
}

#[test]
#[ignore = "dv"]
fn dv_r17_quest_balanced_knobs_step_down_and_back_up_in_e16_order() {
    let apk = apk();
    device();
    install(&apk);
    clear();
    match std::env::var(ROWS_VAR).ok().and_then(|v| v.parse().ok()) {
        Some(rows) => {
            let dir = std::env::temp_dir().join(format!("ftgsxr-r17-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("a temporary directory");
            let file = dir.join("probe.safetensors");
            write_exchange(&file, &every_row_persistent(17, rows)).expect("the model");
            push(&file, "probe.safetensors");
            std::fs::remove_dir_all(&dir).ok();
            println!("r17: the every-row-persistent model, {rows} rows");
        }
        None => {
            let probe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/probes/P1.safetensors");
            push(&probe, "probe.safetensors");
        }
    }
    dv_support::adb(&["logcat", "-G", LOG_BUFFER]);
    let start = launch("r17");
    let result = verdict("r17", start, Duration::from_secs(300));
    let lines = log();
    stop();
    for l in lines.iter().filter(|l| {
        l.contains("r17: ")
            || l.contains("governor step")
            || l.contains("foveation ")
            || l.contains("swapchain ")
            || l.contains("re-creating")
    }) {
        println!("{l}");
    }
    let (pass, detail, _) = result.expect("an r17 verdict line in logcat");
    assert!(pass, "r17 FAIL {detail}");
    if let Err(e) = judge(&lines) {
        panic!("R-17 on the headset: {e}");
    }
}

/// The lines of a run as R-17 expects it, with the launch's lines before the start.
fn canned() -> Vec<String> {
    let mut l = vec![
        "I ftgsxr: swapchain 1680x1760 at render scale 1 (§1.12); the view configuration recommends 1680x1760".to_owned(),
        "I ftgsxr: foveation Medium applied to the swapchain".to_owned(),
        "I ftgsxr: r17: warm-up 30 blocks".to_owned(),
        "I ftgsxr: r17: budget 0.1 ms".to_owned(),
    ];
    for (i, (dir, change, blocks)) in EXPECTED.iter().enumerate() {
        l.push("I ftgsxr: governor step knob=X".to_owned());
        l.push(format!(
            "I ftgsxr: r17: step {} {dir} {change} after {blocks} blocks, GPU total 0.5..0.6 ms",
            i % 6 + 1
        ));
        if i == 4 {
            l.push("I ftgsxr: foveation High applied to the swapchain".to_owned());
        }
        if i == 5 {
            l.push("I ftgsxr: r17: down held 40 blocks without a step after 6 steps, 0 blocks not over; knobs …".to_owned());
        }
        if i == 7 {
            l.push("I ftgsxr: foveation Medium applied to the swapchain".to_owned());
        }
    }
    l.push(
        "I ftgsxr: r17: up held 240 blocks without a step after 6 steps, 0 blocks not under"
            .to_owned(),
    );
    l.push("I ftgsxr: ftgsxr-test: r17 PASS 6 steps down, 6 up".to_owned());
    l
}

/// The driver's judgement (private to it, so a `cpu` test of its own): the expected run
/// passes, and each clause fails on its own.
#[test]
fn judge_passes_e16s_run_and_fails_each_departure() {
    assert_eq!(judge(&canned()), Ok(()));
    // The adapter's layer line at a stepped scale is no re-creation (§1.12).
    let mut l = canned();
    l.insert(
        10,
        "I ftgsxr: splat layer 1512x1584 for a 1680x1760 view at render scale 0.9".to_owned(),
    );
    assert_eq!(judge(&l), Ok(()));
    // Steps on the display's own budget after the verdict are not the run's.
    let mut l = canned();
    l.push("I ftgsxr: governor step knob=RenderScale down=true".to_owned());
    l.push("I ftgsxr: foveation High applied to the swapchain".to_owned());
    assert_eq!(judge(&l), Ok(()));
    let with = |f: &dyn Fn(&mut Vec<String>)| {
        let mut l = canned();
        f(&mut l);
        judge(&l)
    };
    // A count of 11.
    assert!(
        with(&|l| *l = l
            .iter()
            .map(|s| s.replacen("after 10 blocks", "after 11 blocks", 1))
            .collect())
        .is_err()
    );
    // A seventh step down (the sort mode).
    assert!(
        with(&|l| {
            let at = l.iter().position(|s| s.contains("down held")).unwrap();
            l.insert(
                at,
                "I ftgsxr: r17: step 7 down SortMode SharedCenter -> PerView after 10 blocks, GPU"
                    .to_owned(),
            );
        })
        .is_err()
    );
    // A block within the band on the way up.
    assert!(
        with(&|l| *l = l
            .iter()
            .map(|s| s.replace("0 blocks not under", "3 blocks not under"))
            .collect())
        .is_err()
    );
    // A re-created swapchain.
    assert!(
        with(&|l| l.insert(
            10,
            "I ftgsxr: render scale 1 -> 0.9: re-creating the session".to_owned()
        ))
        .is_err()
    );
    assert!(
        with(&|l| l.insert(
            10,
            "I ftgsxr: swapchain 1512x1584 at render scale 0.9 (§1.12)".to_owned()
        ))
        .is_err()
    );
    // The foveation not applied, or applied before its step.
    assert!(with(&|l| l.retain(|s| !s.contains("foveation High applied"))).is_err());
    assert!(
        with(&|l| {
            l.retain(|s| !s.contains("foveation High applied"));
            l.insert(
                5,
                "I ftgsxr: foveation High applied to the swapchain".to_owned(),
            );
        })
        .is_err()
    );
    // A missing governor line.
    assert!(
        with(&|l| {
            let at = l.iter().position(|s| s.contains("governor step")).unwrap();
            l.remove(at);
        })
        .is_err()
    );
}
