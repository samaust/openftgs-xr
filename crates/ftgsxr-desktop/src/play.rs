// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr play` (§1.7 Command line, D-36): the arguments, the settings and the log, then
//! the `App` with the window, `AppPlugin` and, with `--xr`, the XR session (Q-120,
//! Q-121). The end-to-end run and its checks are #289's.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use bevy::app::{App, AppExit};
use bevy::log::LogPlugin;
use bevy::prelude::{DefaultPlugins, PluginGroup};
use bevy::window::{PresentMode, Window, WindowPlugin};
#[cfg(feature = "xr")]
use clap::ArgAction;
use clap::{Arg, ArgMatches, Command, value_parser};
use ftgsxr_app::{
    AppPlugin, ModelRef, Platform, QualityTier, SettingsPath, SettingsRes, SettingsWarnings, Url,
    init_desktop_log, load_settings,
};

use crate::TIERS;
use crate::linux::LinuxPlatform;

/// §1.13's log levels.
const LEVELS: [&str; 5] = ["error", "warn", "info", "debug", "trace"];

/// The suffix that names a scene file (§1.11, Q-157).
const SCENE_SUFFIX: &str = ".ftgsxr-scene.json";

/// The subcommand `play` (§1.7).
pub(crate) fn command() -> Command {
    let cmd = Command::new("play")
        .about("Start the player with the given models, URLs or scene file loaded")
        .arg(
            Arg::new("INPUT")
                .num_args(0..)
                .value_name("MODEL|URL|SCENE")
                .help("A model file or checkpoint directory, an http(s) URL, or one *.ftgsxr-scene.json"),
        )
        .arg(
            Arg::new("tier")
                .long("tier")
                .value_name("TIER")
                .value_parser(TIERS)
                .help("The quality tier for this session (§1.12), not saved"),
        )
        .arg(
            Arg::new("settings")
                .long("settings")
                .value_name("FILE")
                .value_parser(value_parser!(PathBuf))
                .help("Read and save the settings in FILE instead of the default settings.toml"),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .value_name("LEVEL")
                .value_parser(LEVELS)
                .help("The log level for this session (§1.13), not saved"),
        );
    #[cfg(feature = "xr")]
    let cmd = cmd.arg(
        Arg::new("xr")
            .long("xr")
            .action(ArgAction::SetTrue)
            .help("Run the XR session on the system's OpenXR runtime"),
    );
    cmd
}

/// `play`'s arguments sorted (Q-157).
#[derive(Debug, PartialEq)]
pub(crate) struct Inputs {
    /// The models, in the given order.
    pub(crate) models: Vec<ModelRef>,
    /// The scene file.
    pub(crate) scene: Option<PathBuf>,
}

/// Q-157: an argument that parses as an `http` or `https` URL is a URL; a path whose name
/// ends in `.ftgsxr-scene.json` is the scene, of which there is at most one; any other
/// path is a model. A usage error names the second scene; an input error a missing path.
pub(crate) fn classify(args: &[String]) -> Result<Inputs, PlayError> {
    let mut inputs = Inputs {
        models: Vec::new(),
        scene: None,
    };
    for arg in args {
        if let Ok(url) = Url::parse(arg)
            && matches!(url.scheme(), "http" | "https")
        {
            inputs.models.push(ModelRef::Http(url));
            continue;
        }
        let path = PathBuf::from(arg);
        if !path.exists() {
            return Err(PlayError::Input(format!(
                "{arg}: no such file or directory"
            )));
        }
        if arg.ends_with(SCENE_SUFFIX) {
            if let Some(first) = &inputs.scene {
                return Err(PlayError::Usage(format!(
                    "a second scene file {arg} (the first is {}); play takes at most one",
                    first.display()
                )));
            }
            inputs.scene = Some(path);
        } else {
            inputs.models.push(ModelRef::File(path));
        }
    }
    Ok(inputs)
}

/// Why `play` stops before its `App` runs.
#[derive(Debug, PartialEq)]
pub(crate) enum PlayError {
    /// A usage error (clap's form, exit 2).
    Usage(String),
    /// An invalid input or environment (exit 2, §1.7).
    Input(String),
}

/// Runs `play` (§1.7): exit 0 when the window closes, 2 on an invalid input or
/// environment, the app's own code otherwise.
pub(crate) fn run(m: &ArgMatches) -> ExitCode {
    let args: Vec<String> = m
        .get_many::<String>("INPUT")
        .map(|v| v.cloned().collect())
        .unwrap_or_default();
    let inputs = match classify(&args) {
        Ok(inputs) => inputs,
        Err(PlayError::Usage(msg)) => {
            crate::command()
                .error(clap::error::ErrorKind::ArgumentConflict, msg)
                .exit();
        }
        Err(PlayError::Input(msg)) => {
            eprintln!("ftgsxr: {msg}");
            return ExitCode::from(2);
        }
    };
    let platform = match LinuxPlatform::new() {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("ftgsxr: {}: {e}", e.code());
            return ExitCode::from(2);
        }
    };
    // Q-157, Q-170: the settings file, read once here, before the log, for the
    // level and the tier.
    let path = m
        .get_one::<PathBuf>("settings")
        .cloned()
        .unwrap_or_else(|| platform.paths().config.join("settings.toml"));
    let (settings, warnings) = match load_settings(&path) {
        Ok(read) => read,
        Err(e) => (
            ftgsxr_app::Settings::default(),
            vec![format!("{}: {e}", e.code())],
        ),
    };
    let level = m
        .get_one::<String>("log-level")
        .cloned()
        .unwrap_or_else(|| settings.general.log_level.clone());
    if let Err(e) = init_desktop_log(&level, &platform.paths().logs) {
        // Q-170: the player runs on without the log rather than not at all.
        eprintln!("ftgsxr: warning: no log: {}: {e}", e.code());
    }
    for w in &warnings {
        bevy::log::warn!("{}: {w}", path.display());
    }
    let tier_name = m
        .get_one::<String>("tier")
        .cloned()
        .unwrap_or_else(|| settings.general.tier.clone());
    let tier = QualityTier::by_name(&tier_name)
        .or_else(|| QualityTier::by_name("desktop"))
        .expect("invariant: the desktop tier exists");

    let group = DefaultPlugins
        .build()
        .disable::<LogPlugin>()
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: "openftgs-xr".into(),
                // §1.7 Window: vertical sync on by default.
                present_mode: PresentMode::AutoVsync,
                ..Window::default()
            }),
            ..WindowPlugin::default()
        });
    let mut app = App::new();
    // Q-163: the binary's settings, so `--tier` and `--log-level` stay unsaved.
    // Q-224: the load's warnings, for the E-SETTINGS toast.
    app.insert_resource(SettingsPath(path))
        .insert_resource(SettingsRes(settings))
        .insert_resource(SettingsWarnings(warnings));
    if let Err(code) = add_plugins(&mut app, group, m) {
        return code;
    }
    app.add_plugins(AppPlugin {
        platform,
        tier,
        initial: inputs.models,
        scene: inputs.scene,
    });
    exit_code(app.run())
}

/// `DefaultPlugins`, through `add_xr` with `--xr` (Q-120, Q-121); `Err` with exit 2 when
/// `--xr` finds no OpenXR runtime (§1.7, Q-162), E-XR-RUNTIME on standard error.
// Q-162
#[cfg(feature = "xr")]
fn add_plugins(
    app: &mut App,
    group: bevy::app::PluginGroupBuilder,
    m: &ArgMatches,
) -> Result<(), ExitCode> {
    if m.get_flag("xr") {
        let xr = ftgsxr_xr::XrPlugin {
            passthrough: false,
            hands: false,
            sort: ftgsxr_app::SortMode::PerView,
        };
        app.add_plugins(ftgsxr_xr::add_xr(group, xr));
        if let Some(e) = ftgsxr_xr::runtime_unavailable(app) {
            eprintln!("ftgsxr: {}: {}", e.code(), e.user_text());
            return Err(ExitCode::from(2));
        }
    } else {
        app.add_plugins(group);
    }
    Ok(())
}

/// `DefaultPlugins` (no `--xr` without the feature).
#[cfg(not(feature = "xr"))]
fn add_plugins(
    app: &mut App,
    group: bevy::app::PluginGroupBuilder,
    _: &ArgMatches,
) -> Result<(), ExitCode> {
    app.add_plugins(group);
    Ok(())
}

/// The process's code for the app's exit.
fn exit_code(exit: AppExit) -> ExitCode {
    match exit {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ftgsxr-play-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn urls_scenes_and_models_are_told_apart() {
        // Q-157
        let d = dir("classify");
        let model = d.join("a.safetensors");
        let ckpt = d.join("ckpt");
        let scene = d.join("s.ftgsxr-scene.json");
        fs::write(&model, b"").unwrap();
        fs::create_dir(&ckpt).unwrap();
        fs::write(&scene, b"{}").unwrap();
        let args: Vec<String> = [
            model.to_str().unwrap(),
            "https://example.org/m.safetensors",
            scene.to_str().unwrap(),
            "http://h:8000/ckpt/",
            ckpt.to_str().unwrap(),
        ]
        .map(String::from)
        .to_vec();
        let got = classify(&args).unwrap();
        assert_eq!(
            got.models,
            vec![
                ModelRef::File(model),
                ModelRef::Http(Url::parse("https://example.org/m.safetensors").unwrap()),
                ModelRef::Http(Url::parse("http://h:8000/ckpt/").unwrap()),
                ModelRef::File(ckpt),
            ]
        );
        assert_eq!(got.scene, Some(scene));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_second_scene_is_a_usage_error() {
        let d = dir("two-scenes");
        let (a, b) = (d.join("a.ftgsxr-scene.json"), d.join("b.ftgsxr-scene.json"));
        fs::write(&a, b"{}").unwrap();
        fs::write(&b, b"{}").unwrap();
        let args = vec![a.display().to_string(), b.display().to_string()];
        assert!(matches!(classify(&args), Err(PlayError::Usage(_))));
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_missing_path_is_an_input_error_and_another_scheme_a_path() {
        let args = vec!["/nonexistent/m.safetensors".to_owned()];
        assert!(matches!(classify(&args), Err(PlayError::Input(_))));
        // `file:` is not a URL the player fetches: a path, and here a missing one.
        let args = vec!["file:///m.safetensors".to_owned()];
        assert!(matches!(classify(&args), Err(PlayError::Input(_))));
    }
}
