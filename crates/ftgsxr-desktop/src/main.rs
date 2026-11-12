// SPDX-License-Identifier: Apache-2.0

//! The Linux player and its command line, the binary `ftgsxr` (§1.7, §2.1, D-36). M-2 gives
//! it the headless commands: `render` (#118), `info [--json]`, `validate`, `import` and
//! `export` (#117), and `bench` (#122); M-4 `play` and `LinuxPlatform` (#277).

#![warn(missing_docs)]

mod linux;
mod play;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use ftgsxr_app::{
    HeadlessError, QualityTier, cmd_bench, cmd_export, cmd_import, cmd_info, cmd_render,
    cmd_validate, exit_code,
};

/// The player's version, which `info` reports (§1.7).
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The names `--tier` accepts, those of `QualityTier::by_name` (§1.12, Q-89).
pub(crate) const TIERS: [&str; 5] = [
    "conformance",
    "desktop",
    "quest-high",
    "quest-balanced",
    "quest-low",
];

/// The benchmarks the desktop runs (§3.5, Q-103).
// Q-T731.4: B-2, B-7 and B-8 so that `cmd_bench` names their `dv` test.
const BENCHMARKS: [&str; 8] = ["B-1", "B-2", "B-3", "B-5", "B-6", "B-7", "B-8", "B-9"];

/// The exit code of a run whose budgets regressed against the baseline (Q-67).
const REGRESSION_EXIT: u8 = 3;

/// The command line of §1.7.
pub(crate) fn command() -> Command {
    let path = |name: &'static str, help: &'static str| {
        Arg::new(name)
            .required(true)
            .value_parser(value_parser!(PathBuf))
            .help(help)
    };
    Command::new("ftgsxr")
        .version(VERSION)
        .about("openftgs-xr, a player for openftgs models")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(play::command())
        .subcommand(
            Command::new("render")
                .about("Answer a ftgs-render-request headless and write a ftgs-renders file")
                .arg(path(
                    "MODEL",
                    "A ftgs-if file, a checkpoint directory or a PLY snapshot",
                ))
                .arg(path("REQUEST", "The ftgs-render-request JSON file"))
                .arg(path("OUT", "The ftgs-renders file to write"))
                .arg(
                    Arg::new("tier")
                        .long("tier")
                        .value_name("TIER")
                        .value_parser(TIERS)
                        .default_value("conformance")
                        .help("The quality tier (§1.12)"),
                ),
        )
        .subcommand(
            Command::new("info")
                .about("Print the adapter reply of openftgs's harness contract")
                .arg(
                    Arg::new("json")
                        .long("json")
                        .action(ArgAction::SetTrue)
                        .help("Print the reply as one JSON object"),
                ),
        )
        .subcommand(
            Command::new("validate")
                .about("Read and validate each model and print its header information")
                .arg(
                    Arg::new("MODEL")
                        .required(true)
                        .num_args(1..)
                        .value_parser(value_parser!(PathBuf))
                        .help("A ftgs-if file, a checkpoint directory or a PLY snapshot"),
                ),
        )
        .subcommand(
            Command::new("import")
                .about("Validate a model and write it as a ftgs-if file")
                .arg(path("IF", "The model to import"))
                .arg(path("OUT", "The ftgs-if file to write")),
        )
        .subcommand(
            Command::new("export")
                .about("Validate a model and write it as a ftgs-if file")
                .arg(path("MODEL", "The model to export"))
                .arg(path("OUT", "The ftgs-if file to write")),
        )
        .subcommand(
            Command::new("bench")
                .about("Run a benchmark of §3.5 and compare it with the stored baseline")
                .arg(
                    Arg::new("NAME")
                        .required(true)
                        .value_parser(BENCHMARKS)
                        .help("The benchmark: B-1, B-3, B-5, B-6 or B-9 (B-2, B-7, B-8 run on the headset)"),
                )
                .arg(
                    Arg::new("out")
                        .long("out")
                        .value_name("BENCH.json")
                        .value_parser(value_parser!(PathBuf))
                        .help("Write the benchmark file here instead of standard output"),
                )
                .arg(
                    Arg::new("model")
                        .long("model")
                        .value_name("MODEL")
                        .value_parser(value_parser!(PathBuf))
                        .help("Run on this model instead of the synthetic ones"),
                ),
        )
}

/// The path argument `name` of `m`, which clap has made required.
fn path_arg<'a>(m: &'a ArgMatches, name: &str) -> &'a Path {
    m.get_one::<PathBuf>(name)
        .expect("invariant: clap requires the argument")
}

/// Runs the subcommand, printing its output to standard output (Q-77); `Ok(false)` when
/// a benchmark's budgets regressed (exit 3, Q-67).
fn run(matches: &ArgMatches) -> Result<bool, HeadlessError> {
    match matches.subcommand() {
        Some(("render", m)) => {
            // Q-89: clap accepts only the names of `TIERS`.
            let name = m
                .get_one::<String>("tier")
                .expect("invariant: the tier has a default");
            let tier = QualityTier::by_name(name).expect("invariant: clap checks the name");
            cmd_render(
                path_arg(m, "MODEL"),
                path_arg(m, "REQUEST"),
                path_arg(m, "OUT"),
                tier,
                VERSION,
            )?;
        }
        Some(("info", m)) => println!("{}", cmd_info(VERSION, m.get_flag("json"))),
        Some(("validate", m)) => {
            let models: Vec<PathBuf> = m
                .get_many::<PathBuf>("MODEL")
                .expect("invariant: clap requires a model")
                .cloned()
                .collect();
            // Q-77: one model per call, so each valid model's line is printed as it is
            // read and the first invalid one stops the run (§1.7).
            for path in &models {
                let info = cmd_validate(std::slice::from_ref(path))?
                    .pop()
                    .expect("invariant: cmd_validate returns one ModelInfo per model");
                // Q-110, D-39: E6's conventions rendered as openftgs's instead.
                for w in info.conventions.warnings() {
                    eprintln!("ftgsxr: warning: {}: {w}", path.display());
                }
                // `ModelKind` is the core's; the desktop crate depends on the app only
                // (§2.1), so the kind is told by its variant's name.
                let kind = format!("{:?}", info.kind);
                let kind = if kind.starts_with("Checkpoint") {
                    "ftgs-checkpoint"
                } else if kind.starts_with("Ply") {
                    "ply"
                } else {
                    "ftgs-if"
                };
                let producer = if info.producer.is_empty() {
                    "(none)"
                } else {
                    &info.producer
                };
                println!(
                    "{}: kind {kind}, N {}, SH degree {}, tau0 {:?}, tau1 {:?}, time_unit {}, producer {producer}, {} bytes",
                    path.display(),
                    info.n,
                    info.sh_degree,
                    info.time.tau0,
                    info.time.tau1,
                    info.time.time_unit,
                    info.file_bytes,
                );
            }
        }
        Some(("import", m)) => cmd_import(path_arg(m, "IF"), path_arg(m, "OUT"))?,
        Some(("export", m)) => cmd_export(path_arg(m, "MODEL"), path_arg(m, "OUT"))?,
        Some(("bench", m)) => {
            // Q-67, Q-103: the file at --out or on standard output; failed budgets and
            // regressions on standard error; exit 3 when any budget regressed.
            let name = m
                .get_one::<String>("NAME")
                .expect("invariant: clap requires the name");
            let out = m.get_one::<PathBuf>("out").map(PathBuf::as_path);
            let model = m.get_one::<PathBuf>("model").map(PathBuf::as_path);
            let report = cmd_bench(name, out, model)?;
            if out.is_none() {
                print!("{}", report.json);
            }
            for note in stderr_notes(&report.json) {
                eprintln!("ftgsxr: {name}: {note}");
            }
            // Q-103: the form `report_json_puts_a_budget_not_met_on_a_pass_false_line`
            // (ftgsxr-app's bench) pins (#198).
            for line in report.json.lines() {
                if line.trim_start().starts_with("\"pass\": false") {
                    eprintln!("ftgsxr: {name}: a budget is not met (see \"pass\": false)");
                    break;
                }
            }
            for line in &report.regressions {
                eprintln!("ftgsxr: regression: {line}");
            }
            return Ok(report.regressions.is_empty());
        }
        _ => unreachable!("invariant: clap requires a known subcommand"),
    }
    Ok(true)
}

/// The starts of the `notes` the library logs and the binary prints on standard error
/// (Q-110's split, #193): a baseline not compared (Q-103) and GPU validation messages
/// (Q-116).
// Q-103, Q-116
const STDERR_NOTES: [&str; 2] = ["baseline not compared: ", "GPU validation reported "];

/// The notes of a benchmark file's text that go to standard error, unescaped: each is a
/// string member of `notes` on a line of its own (Q-103's two-space indented form).
fn stderr_notes(json: &str) -> Vec<String> {
    json.lines()
        .filter_map(|line| {
            let text = line.trim().strip_suffix(',').unwrap_or(line.trim());
            let inner = text.strip_prefix('"')?.strip_suffix('"')?;
            STDERR_NOTES
                .iter()
                .any(|p| inner.starts_with(p))
                .then(|| unescape(inner))
        })
        .collect()
}

/// A JSON string's text with its escapes `\"`, `\\`, `\n` and `\t` undone; any other
/// escape is kept as written.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// The entry point (§1.7): exit 0 on success, `exit_code`'s 2 or 3 on a failure, 3 on a
/// benchmark's regression (Q-67), clap's 2 on a usage error.
// Q-77, Q-67
fn main() -> ExitCode {
    let matches = command().get_matches();
    if let Some(("play", m)) = matches.subcommand() {
        return play::run(m);
    }
    match run(&matches) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(REGRESSION_EXIT),
        Err(e) => {
            eprintln!("ftgsxr: {e}");
            ExitCode::from(u8::try_from(exit_code(&e)).unwrap_or(2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stderr_notes_picks_the_baseline_and_validation_notes() {
        // #193, Q-103, Q-116: of the notes, only these two kinds reach standard error.
        let json = "{\n  \"notes\": [\n    \"a frame's time is the wall time (Q-102)\",\n    \"GPU validation reported 1 messages (Q-116): say \\\"x\\\"\",\n    \"baseline not compared: profile debug against release (Q-103)\"\n  ]\n}\n";
        assert_eq!(
            stderr_notes(json),
            vec![
                "GPU validation reported 1 messages (Q-116): say \"x\"".to_owned(),
                "baseline not compared: profile debug against release (Q-103)".to_owned(),
            ]
        );
    }
}
