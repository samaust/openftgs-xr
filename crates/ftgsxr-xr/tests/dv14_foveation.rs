// SPDX-License-Identifier: Apache-2.0

//! Dv-14 (§3.4, `dv` lane; §1.12 Headset display, D-58, E16's floor; #810), on Q-454's
//! setup: the driver writes Q-420's every-row-persistent model at 5 000 rows, pushes it
//! as `dv/spike.safetensors`, launches the runner's `dv14` (spike S-F1's runner held
//! 1.0 m ahead of the left eye) with the debug APK's validation layer, and reads the
//! `dv14: window` lines of `display::dv14`: at render scale 1.0 and at 0.6, four rounds
//! of off and high. The host judges (Q-122): a window is kept only in the fast mode,
//! its sort median at most 1.2 × the lowest among the run's windows; a level's figure is
//! the mean of its kept windows' `draw` medians, which span both eyes' splat passes in
//! shared-center mode; at each scale high's must be at least 20 % below off's. A scale
//! where a level has no kept window fails as incomplete. Run it alone, with
//! `--test-threads 1` and `--nocapture` for the table. Prerequisites: `dv_support`'s.
// Q-454, Q-457, Q-122

use crate::dv_spikes::every_row_persistent;
use crate::dv_support;

use std::time::Duration;

use dv_support::{
    DebugLayers, Validation, apk, clear, device, install, launch, log, push, stop, verdict,
};
use ftgsxr_core::write_exchange;

/// Q-454's rows.
const ROWS: usize = 5_000;
/// Dv-14's margin (§3.4): high at least 20 % below off.
const MARGIN: f32 = 0.20;
/// The fast mode: a sort median at most this times the run's lowest (Q-454).
const FAST: f32 = 1.2;
const SCALES: [&str; 2] = ["1.0", "0.6"];

/// One window's line.
#[derive(Clone, Debug, PartialEq)]
struct Window {
    scale: String,
    level: String,
    round: usize,
    draw: f32,
    sort: f32,
}

/// `dv14: window <scale> <level> r<round> draw median <ms> ms, … sort <ms> over …`.
fn parse(line: &str) -> Option<Window> {
    let rest = line.split_once("dv14: window ")?.1;
    let w: Vec<&str> = rest.split_whitespace().collect();
    let num = |key: &str| {
        let i = w.iter().position(|x| *x == key)?;
        w.get(i + 1)?.parse::<f32>().ok()
    };
    Some(Window {
        scale: w.first()?.to_string(),
        level: w.get(1)?.to_string(),
        round: w.get(2)?.strip_prefix('r')?.parse().ok()?,
        draw: num("median")?,
        sort: num("sort")?,
    })
}

/// A scale's outcome.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// Off's and high's figures, ms, and high's saving.
    Judged { off: f32, high: f32, saving: f32 },
    /// A level with no kept window.
    Incomplete(String),
}

/// The report: each window with whether it was kept, and each scale's outcome.
struct Report {
    rows: Vec<(Window, bool)>,
    scales: Vec<(String, Outcome)>,
}

impl Report {
    fn pass(&self) -> bool {
        self.scales
            .iter()
            .all(|(_, o)| matches!(o, Outcome::Judged { saving, .. } if *saving >= MARGIN - 1e-6))
    }
}

fn judge(windows: &[Window]) -> Report {
    let lowest = windows.iter().map(|w| w.sort).fold(f32::INFINITY, f32::min);
    let rows: Vec<(Window, bool)> = windows
        .iter()
        .map(|w| (w.clone(), w.sort <= FAST * lowest))
        .collect();
    let figure = |scale: &str, level: &str| {
        let kept: Vec<f32> = rows
            .iter()
            .filter(|(w, k)| *k && w.scale == scale && w.level == level)
            .map(|(w, _)| w.draw)
            .collect();
        (!kept.is_empty()).then(|| kept.iter().sum::<f32>() / kept.len() as f32)
    };
    let scales = SCALES
        .iter()
        .map(|&s| {
            let o = match (figure(s, "Off"), figure(s, "High")) {
                (Some(off), Some(high)) => Outcome::Judged {
                    off,
                    high,
                    saving: 1.0 - high / off,
                },
                (None, _) => Outcome::Incomplete("no kept window at Off".into()),
                (_, None) => Outcome::Incomplete("no kept window at High".into()),
            };
            (s.to_owned(), o)
        })
        .collect();
    Report { rows, scales }
}

#[test]
#[ignore = "dv"]
fn dv_dv14_high_foveation_draws_at_least_20_percent_below_off_at_both_scales() {
    let apk = apk();
    device();
    install(&apk);
    let dir = std::env::temp_dir().join(format!("ftgsxr-dv14-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    let file = dir.join("spike.safetensors");
    write_exchange(&file, &every_row_persistent(1, ROWS)).expect("the model is written");
    clear();
    push(&file, "spike.safetensors");
    std::fs::remove_dir_all(&dir).ok();
    dv_support::adb(&["logcat", "-G", "16M"]);
    let layers = DebugLayers::enable();
    let start = launch("dv14");
    // 16 windows of 90 blocks, slow-mode windows included.
    let result = verdict("dv14", start, Duration::from_secs(900));
    let lines = log();
    let validation = Validation::read();
    stop();
    drop(layers);
    println!(
        "--- validation: layer announced {}, {} VALIDATION lines ---",
        validation.announced,
        validation.messages.len()
    );
    for m in validation.messages.iter().take(20) {
        println!("{m}");
    }
    for l in lines
        .iter()
        .filter(|l| l.contains("layer foveation") || l.contains("density map support"))
    {
        println!("{l}");
    }
    let (pass, detail, _) = result.expect("a dv14 verdict line in logcat");
    assert!(pass, "dv14 FAIL {detail}");
    let windows: Vec<Window> = lines.iter().filter_map(|l| parse(l)).collect();
    assert_eq!(windows.len(), 16, "every window's line");
    let report = judge(&windows);
    println!("--- Dv-14 ({ROWS} rows, 1.0 m) ---");
    for (w, kept) in &report.rows {
        println!(
            "{} {:<4} r{} draw {:.3} sort {:.3} {}",
            w.scale,
            w.level,
            w.round,
            w.draw,
            w.sort,
            if *kept { "kept" } else { "slow" }
        );
    }
    for (s, o) in &report.scales {
        println!("scale {s}: {o:?}");
    }
    assert!(report.pass(), "Dv-14: {:?}", report.scales);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(scale: &str, level: &str, draw: f32, sort: f32) -> Window {
        Window {
            scale: scale.into(),
            level: level.into(),
            round: 0,
            draw,
            sort,
        }
    }

    #[test]
    fn parse_reads_a_window_line() {
        let l = "11-16 10:00:00.000 I ftgsxr: dv14: window 0.6 High r3 draw median 7.5 ms, \
                 p10 7 p90 8 min 6 max 9, total median 9 cull 0.4 sort 1.25 over 90 blocks \
                 in 1.20 s, visible 5000..5000, overflow false";
        assert_eq!(
            parse(l),
            Some(Window {
                scale: "0.6".into(),
                level: "High".into(),
                round: 3,
                draw: 7.5,
                sort: 1.25,
            })
        );
        assert_eq!(parse("dv14: render scale 1, foveation High"), None);
    }

    #[test]
    fn judge_keeps_fast_windows_and_averages_their_draws() {
        // Lowest sort 1.0: 1.2 is kept (inclusive), 1.3 is slow and left out.
        let r = judge(&[
            w("1.0", "Off", 10.0, 1.0),
            w("1.0", "Off", 12.0, 1.2),
            w("1.0", "Off", 20.0, 1.3),
            w("1.0", "High", 8.0, 1.1),
            w("0.6", "Off", 5.0, 1.0),
            w("0.6", "High", 4.0, 1.0),
        ]);
        let kept: Vec<bool> = r.rows.iter().map(|(_, k)| *k).collect();
        assert_eq!(kept, [true, true, false, true, true, true]);
        // Off at 1.0: mean of 10 and 12 = 11; high 8: saving 1 - 8/11.
        assert_eq!(
            r.scales[0].1,
            Outcome::Judged {
                off: 11.0,
                high: 8.0,
                saving: 1.0 - 8.0 / 11.0
            }
        );
        // 0.6: exactly 20 % passes.
        assert!(
            matches!(r.scales[1].1, Outcome::Judged { saving, .. } if (saving - 0.2).abs() < 1e-6)
        );
        assert!(r.pass());
    }

    #[test]
    fn judge_fails_a_saving_below_the_margin() {
        let r = judge(&[
            w("1.0", "Off", 10.0, 1.0),
            w("1.0", "High", 8.5, 1.0),
            w("0.6", "Off", 10.0, 1.0),
            w("0.6", "High", 5.0, 1.0),
        ]);
        assert!(!r.pass());
    }

    #[test]
    fn judge_reports_a_level_without_kept_windows_as_incomplete() {
        let r = judge(&[
            w("1.0", "Off", 10.0, 1.0),
            w("1.0", "High", 5.0, 2.0),
            w("0.6", "Off", 10.0, 1.0),
            w("0.6", "High", 5.0, 1.0),
        ]);
        assert_eq!(
            r.scales[0].1,
            Outcome::Incomplete("no kept window at High".into())
        );
        assert!(!r.pass());
    }
}
