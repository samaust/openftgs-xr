// SPDX-License-Identifier: Apache-2.0

//! The headset's benchmark runner (§3.5, Q-406): the launch intent's string extra
//! `ftgsxr-bench` names a benchmark that runs before the XR session on the headless
//! device, as Dv-3 does (Q-122): B-3, B-5 or B-6 through `ftgsxr_app::cmd_bench_on`, with
//! the headset's build fingerprint from the extra `ftgsxr-bench-headset` (Q-418). It
//! writes `dv/<NAME>.json` and prints `ftgsxr-bench: <NAME> DONE|FAIL <detail>` under the
//! tag `ftgsxr`.

use std::path::Path;

/// The intent extra naming the benchmark (Q-406).
#[cfg(target_os = "android")]
pub(crate) const EXTRA: &str = "ftgsxr-bench";
/// The intent extra holding the headset's build fingerprint (Q-406, Q-418).
// Q-418
#[cfg(target_os = "android")]
pub(crate) const HEADSET_EXTRA: &str = "ftgsxr-bench-headset";
/// The `machine.headset` of a launch without `HEADSET_EXTRA`: a field the device cannot
/// give (Q-417).
// Q-417
const UNKNOWN: &str = "unknown";

/// Runs the benchmark `name` with its file at `dv/<NAME>.json`: `Ok` with the detail of
/// the DONE line (the file and any regression against a stored baseline), `Err` with
/// the error's `<code>: <text>`.
// Q-406
pub(crate) fn run(name: &str, dv: &Path, headset: Option<&str>) -> Result<String, String> {
    std::fs::create_dir_all(dv).map_err(|e| format!("E-IO: {}: {e}", dv.display()))?;
    let out = dv.join(format!("{name}.json"));
    let report = ftgsxr_app::cmd_bench_on(name, Some(&out), None, Some(headset.unwrap_or(UNKNOWN)))
        .map_err(|e| e.to_string())?;
    let mut detail = format!("dv/{name}.json");
    if !report.regressions.is_empty() {
        detail.push_str(&format!(
            "; {} regressions: {}",
            report.regressions.len(),
            report.regressions.join("; ")
        ));
    }
    Ok(detail)
}

/// The runner's logcat line (Q-406).
pub(crate) fn line(name: &str, result: &Result<String, String>) -> String {
    let (word, detail) = match result {
        Ok(d) => ("DONE", d),
        Err(d) => ("FAIL", d),
    };
    format!("ftgsxr-bench: {name} {word} {detail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_writes_done_or_fail_with_the_detail() {
        assert_eq!(
            line("B-3", &Ok("dv/B-3.json".to_owned())),
            "ftgsxr-bench: B-3 DONE dv/B-3.json"
        );
        assert_eq!(
            line("B-2", &Err("E-FILE-FORMAT: B-2: no".to_owned())),
            "ftgsxr-bench: B-2 FAIL E-FILE-FORMAT: B-2: no"
        );
    }

    #[test]
    fn run_refuses_a_benchmark_the_headset_does_not_run_before_the_session() {
        // Q-406: B-1 is the desktop's; the refusal comes before any GPU work.
        let dv = std::env::temp_dir().join(format!("ftgsxr-bench-runner-{}", std::process::id()));
        let e = run("B-1", &dv, Some("print")).expect_err("refused");
        assert!(e.contains("not a headset benchmark"), "{e}");
        assert!(!dv.join("B-1.json").exists());
        let _ = std::fs::remove_dir_all(&dv);
    }
}
