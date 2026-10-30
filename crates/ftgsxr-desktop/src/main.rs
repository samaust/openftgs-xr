// SPDX-License-Identifier: Apache-2.0

//! The Linux player and its command line, the binary `ftgsxr` (§1.7, §2.1, D-36). M-2 gives
//! it the headless commands: `render` (#118), and `info [--json]`, `validate`, `import` and
//! `export` (#117); `play` and `bench` arrive with their milestones.

#![warn(missing_docs)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use ftgsxr_app::{
    HeadlessError, QualityTier, cmd_export, cmd_import, cmd_info, cmd_render, cmd_validate,
    exit_code,
};

/// The player's version, which `info` reports (§1.7).
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The names `--tier` accepts, those of `QualityTier::by_name` (§1.12, Q-89).
const TIERS: [&str; 5] = [
    "conformance",
    "desktop",
    "quest-high",
    "quest-balanced",
    "quest-low",
];

/// The command line of §1.7, the subcommands implemented so far.
fn command() -> Command {
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
}

/// The path argument `name` of `m`, which clap has made required.
fn path_arg<'a>(m: &'a ArgMatches, name: &str) -> &'a Path {
    m.get_one::<PathBuf>(name)
        .expect("invariant: clap requires the argument")
}

/// Runs the subcommand, printing its output to standard output (Q-77).
fn run(matches: &ArgMatches) -> Result<(), HeadlessError> {
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
            for (path, info) in models.iter().zip(cmd_validate(&models)?) {
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
        _ => unreachable!("invariant: clap requires a known subcommand"),
    }
    Ok(())
}

/// The entry point (§1.7): exit 0 on success, `exit_code`'s 2 or 3 on a failure, clap's 2
/// on a usage error.
// Q-77
fn main() -> ExitCode {
    let matches = command().get_matches();
    match run(&matches) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ftgsxr: {e}");
            ExitCode::from(u8::try_from(exit_code(&e)).unwrap_or(2))
        }
    }
}
