// SPDX-License-Identifier: Apache-2.0

//! The settings file `settings.toml` (`ftgsxr-settings` v1, §1.11, D-23): one struct per
//! table with §1's defaults, a lenient reader (unknown keys and wrong-type keys warn and
//! take the default; an unknown major version or an unreadable file is set aside) and an
//! atomic writer.

use std::collections::BTreeMap;
use std::path::Path;

use ftgsxr_core::LoopMode;
use toml::{Table, Value};

use crate::util::{set_aside, write_atomic};
use crate::{IoError, ListingFlavor, SourceSpec};

/// The file's `format` (§1.11).
const FORMAT: &str = "ftgsxr-settings";
/// The major version this reader knows and the writer writes (§1.11).
const VERSION: i64 = 1;
/// The top-level keys of `ftgsxr-settings` v1 (§1.11).
const TOP: &[&str] = &[
    "format",
    "version",
    "general",
    "sources",
    "playback",
    "display",
    "environment",
    "locomotion",
    "audio",
    "desktop",
    "xr",
    "cache",
];

/// §1.12's tiers.
const TIERS: &[&str] = &[
    "conformance",
    "desktop",
    "quest-high",
    "quest-balanced",
    "quest-low",
];
/// §1.13's log levels.
const LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];
/// §1.6's loop modes.
pub(crate) const LOOP_MODES: &[&str] = &["loop", "ping-pong", "once"];
/// §1.5's render modes.
pub(crate) const RENDER_MODES: &[&str] = &["splats", "centers", "both", "hidden"];
/// §1.4's colormaps (Q-50).
const COLORMAPS: &[&str] = &["dc", "time", "speed", "opacity"];
/// §1.9's environment modes (Q-50).
pub(crate) const ENVIRONMENT_MODES: &[&str] = &["vr", "passthrough"];
/// §1.9's lighting modes (Q-50).
pub(crate) const LIGHTING_MODES: &[&str] = &["off", "manual", "auto-global", "auto-local"];
/// E12's snap angles in degrees.
const SNAP_ANGLES: &[f64] = &[30.0, 45.0, 90.0];
/// §1.12's foveation levels (D-27).
const FOVEATION: &[&str] = &["off", "low", "medium", "high", "dynamic"];
/// D-24's reference spaces.
const SPACES: &[&str] = &["stage", "local-floor", "local"];
/// §1.11's source kinds.
pub(crate) const SOURCE_KINDS: &[&str] = &["dir", "http", "documents"];

/// E14's manual mode (§1.9): exposure in EV in \[−4, 4\], white-balance temperature and
/// tint in \[−1, 1\]; zero is the identity transform (Q-50).
#[derive(Clone, Debug, PartialEq)]
pub struct ManualLighting {
    /// g = 2^EV (E14).
    pub exposure_ev: f32,
    /// The temperature slider t (E14).
    pub temperature: f32,
    /// The tint slider u (E14).
    pub tint: f32,
}

/// The environment settings of §1.9, the `[environment]` table of `settings.toml`
/// (§1.11) and a scene file's `environment` (§1.5). In `settings.toml`, `background` is
/// the `[display]` table's key (Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentSettings {
    /// `vr` or `passthrough` (§1.9, Q-50).
    pub mode: String,
    /// VR mode's background color, sRGB-encoded in \[0, 1\] (§1.4).
    pub background: [f32; 3],
    /// The passthrough style's brightness in \[−1, 1\] (§1.9).
    pub passthrough_brightness: f32,
    /// The passthrough style's contrast in \[−1, 1\] (§1.9).
    pub passthrough_contrast: f32,
    /// The passthrough style's saturation in \[−1, 1\] (§1.9).
    pub passthrough_saturation: f32,
    /// Depth occlusion in passthrough mode (§1.9, E13).
    pub occlusion: bool,
    /// Soft rather than hard occlusion (E13, D-16).
    pub occlusion_soft: bool,
    /// E13's δ in metres, in \[0, 0.2\] (§1.9, D-16).
    pub occlusion_delta_m: f32,
    /// The depth provider's hand removal (§1.9, D-16).
    pub hand_removal: bool,
    /// `off`, `manual`, `auto-global` or `auto-local` (§1.9, Q-50).
    pub lighting_mode: String,
    /// The manual mode's sliders (E14).
    pub lighting_manual: ManualLighting,
    /// E14's strength γ in \[0, 1\].
    pub lighting_gamma: f32,
    /// E14's reference E\_ref (Ȳ, r̄, ḡ, b̄) once captured.
    pub lighting_reference: Option<[f32; 4]>,
    /// The contact shadows of §1.9.
    pub contact_shadows: bool,
}

/// The per-sound switches of §1.10, the `[audio.sounds]` table (Q-49).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoundSwitches {
    /// The hover tick.
    pub hover: bool,
    /// The select sound.
    pub select: bool,
    /// The grab start and end sounds.
    pub grab: bool,
    /// The snap-turn sound.
    pub snap_turn: bool,
    /// The UI click.
    pub click: bool,
    /// The error tones.
    pub error: bool,
    /// The confirmation ("pose saved").
    pub confirm: bool,
}

/// `[general]` (§1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralSettings {
    /// The quality tier's name (§1.12).
    pub tier: String,
    /// The quality governor (E16).
    pub governor: bool,
    /// The UI scale in \[0.8, 1.5\] (§1.8).
    pub ui_scale: f32,
    /// `error`, `warn`, `info`, `debug` or `trace` (§1.13).
    pub log_level: String,
}

/// `[playback]` (§1.6, §1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackSettings {
    /// The clock's speed, in E9's \[−4, 4\] \\ {0} (Q-50).
    pub speed: f64,
    /// The key `loop`, spelled `loop`, `ping-pong` or `once` (§1.6, Q-50).
    pub loop_mode: LoopMode,
}

/// `[display.overlays]`: §1.4's optional overlays, named as §2.2's `OverlayToggle`
/// (Q-49).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlaySettings {
    /// The bounding box at the current time.
    pub boxes: bool,
    /// Velocity vectors.
    pub velocities: bool,
    /// The velocity vectors' k: one primitive in k.
    pub velocity_every: u32,
    /// The time-slice tint.
    pub time_tint: bool,
}

/// `[display]` (§1.4, §1.11, Q-49); its `background` key is
/// `EnvironmentSettings::background`.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplaySettings {
    /// The render mode: `splats`, `centers`, `both` or `hidden` (§1.5).
    pub mode: String,
    /// The centers' radius ρ in pixels, in \[1, 8\] (§1.4).
    pub centers_radius: f32,
    /// The centers' colormap: `dc`, `time`, `speed` or `opacity` (§1.4, Q-50).
    pub colormap: String,
    /// The floor grid at y = 0 (§1.4).
    pub grid: bool,
    /// The optional overlays (§1.4).
    pub overlays: OverlaySettings,
}

/// `[locomotion]` (§1.8, E12, §1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct LocomotionSettings {
    /// E12's v\_max in m/s, in \[1, 4\].
    pub speed_mps: f32,
    /// Snap rather than smooth turn.
    pub snap_turn: bool,
    /// E12's θ\_snap in degrees: 30, 45 or 90.
    pub snap_deg: f32,
    /// E12's ω in degrees per second.
    pub smooth_dps: f32,
    /// The vignette during smooth motion.
    pub vignette: bool,
    /// Teleport instead of the smooth move.
    pub teleport: bool,
    /// The seated height offset in metres, in \[−0.5, 0.5\].
    pub height_offset_m: f32,
    /// Left-handed mode.
    pub left_handed: bool,
    /// Reduced motion: no animated transitions.
    pub reduced_motion: bool,
}

/// `[audio]` (§1.10, §1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSettings {
    /// The master mute.
    pub muted: bool,
    /// The master volume in percent, \[0, 100\].
    pub master: f32,
    /// The UI volume in percent, \[0, 100\].
    pub ui: f32,
    /// The soundtrack volume in percent, \[0, 100\].
    pub soundtrack: f32,
    /// Spatialized soundtracks on the headset.
    pub spatial: bool,
    /// The per-sound switches.
    pub sounds: SoundSwitches,
}

/// `[desktop]` (§1.7, Appendix D.1, §1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct DesktopSettings {
    /// The vertical field of view in degrees, in \[40, 110\].
    pub fov_deg: f32,
    /// The fly camera's base speed in m/s, in \[0.1, 20\].
    pub fly_speed_mps: f32,
    /// Rebindings, action → key; an action not named keeps Appendix D.1's binding.
    pub keys: BTreeMap<String, String>,
}

/// `[xr]` (§1.8, §1.12, D-24, D-27, §1.11, Q-49).
#[derive(Clone, Debug, PartialEq)]
pub struct XrSettings {
    /// The display refresh rate in Hz; `None` is the runtime's default.
    pub refresh_rate: Option<f32>,
    /// `off`, `low`, `medium`, `high` or `dynamic`; `None` is the tier's.
    pub foveation: Option<String>,
    /// The render scale; `None` is the tier's.
    pub render_scale: Option<f32>,
    /// The reference spaces D-24's chain may try, in order, from `stage`, `local-floor`
    /// and `local`, stored as written; the chain always ends at LOCAL, with a warning,
    /// even when the list leaves `local` out, which `ftgsxr_xr::session` applies (Q-49).
    pub reference_space: Vec<String>,
}

/// `[cache]` (§1.5, §1.11).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSettings {
    /// The download cache's size limit in bytes.
    pub max_bytes: u64,
}

/// The whole `settings.toml` (§1.11, D-23).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Settings {
    /// `[general]`.
    pub general: GeneralSettings,
    /// `[[sources]]`, the configured model sources (§1.5); none by default (Q-49).
    pub sources: Vec<SourceSpec>,
    /// `[playback]`.
    pub playback: PlaybackSettings,
    /// `[display]`.
    pub display: DisplaySettings,
    /// `[environment]` (and `[display] background`).
    pub environment: EnvironmentSettings,
    /// `[locomotion]`.
    pub locomotion: LocomotionSettings,
    /// `[audio]`.
    pub audio: AudioSettings,
    /// `[desktop]`.
    pub desktop: DesktopSettings,
    /// `[xr]`.
    pub xr: XrSettings,
    /// `[cache]`.
    pub cache: CacheSettings,
}

impl Default for ManualLighting {
    // Q-50
    fn default() -> Self {
        ManualLighting {
            exposure_ev: 0.0,
            temperature: 0.0,
            tint: 0.0,
        }
    }
}

impl Default for EnvironmentSettings {
    // Q-50
    fn default() -> Self {
        EnvironmentSettings {
            mode: "vr".into(),
            background: [0.1, 0.1, 0.1],
            passthrough_brightness: 0.0,
            passthrough_contrast: 0.0,
            passthrough_saturation: 0.0,
            occlusion: true,
            occlusion_soft: true,    // D-16
            occlusion_delta_m: 0.05, // D-16
            hand_removal: false,     // D-16
            lighting_mode: "off".into(),
            lighting_manual: ManualLighting::default(),
            lighting_gamma: 0.5,
            lighting_reference: None,
            contact_shadows: true,
        }
    }
}

impl Default for SoundSwitches {
    // Q-50
    fn default() -> Self {
        SoundSwitches {
            hover: true,
            select: true,
            grab: true,
            snap_turn: true,
            click: true,
            error: true,
            confirm: true,
        }
    }
}

impl Default for GeneralSettings {
    // Q-50
    fn default() -> Self {
        let tier = if cfg!(target_os = "android") {
            "quest-balanced"
        } else {
            "desktop"
        };
        GeneralSettings {
            tier: tier.into(),
            governor: true,
            ui_scale: 1.0,
            log_level: "info".into(),
        }
    }
}

impl Default for PlaybackSettings {
    // Q-49
    fn default() -> Self {
        PlaybackSettings {
            speed: 1.0,
            loop_mode: LoopMode::Loop,
        }
    }
}

impl Default for OverlaySettings {
    // Q-49, Q-50
    fn default() -> Self {
        OverlaySettings {
            boxes: false,
            velocities: false,
            velocity_every: 64,
            time_tint: false,
        }
    }
}

impl Default for DisplaySettings {
    // Q-50
    fn default() -> Self {
        DisplaySettings {
            mode: "splats".into(),
            centers_radius: 3.0,
            colormap: "dc".into(),
            grid: true,
            overlays: OverlaySettings::default(),
        }
    }
}

impl Default for LocomotionSettings {
    // D-44
    fn default() -> Self {
        LocomotionSettings {
            speed_mps: 2.0,
            snap_turn: true,
            snap_deg: 45.0,
            smooth_dps: 60.0,
            vignette: true,
            teleport: false,
            height_offset_m: 0.0,
            left_handed: false,
            reduced_motion: false,
        }
    }
}

impl Default for AudioSettings {
    // Q-50
    fn default() -> Self {
        AudioSettings {
            muted: false,
            master: 100.0,
            ui: 100.0,
            soundtrack: 100.0,
            spatial: true,
            sounds: SoundSwitches::default(),
        }
    }
}

impl Default for DesktopSettings {
    // Q-49
    fn default() -> Self {
        DesktopSettings {
            fov_deg: 60.0,
            fly_speed_mps: 2.0,
            keys: BTreeMap::new(),
        }
    }
}

impl Default for XrSettings {
    // Q-49, Q-50, D-24
    fn default() -> Self {
        XrSettings {
            refresh_rate: None,
            foveation: None,
            render_scale: None,
            reference_space: vec!["stage".into(), "local-floor".into(), "local".into()],
        }
    }
}

impl Default for CacheSettings {
    fn default() -> Self {
        CacheSettings {
            max_bytes: 20 << 30,
        }
    }
}

/// Reads `settings.toml` at `path` (§1.11, D-23): the settings and the warnings to show
/// and log, each naming the key by its dotted path.
///
/// A missing file gives the defaults and no warning. A missing key takes its default; an
/// unknown key is ignored with a warning; a key of the wrong type, or whose value lies
/// outside its range or spelling, takes its default with a warning (Q-50). A file of
/// an unknown major version, or one that cannot be read as `ftgsxr-settings`, is renamed
/// to `<file name>.unknown-<date>` beside it and the defaults load, with an E-SETTINGS
/// warning (Q-51). E-IO when the file cannot be read or set aside.
// Q-49, Q-50, Q-51
pub fn load_settings(path: &Path) -> Result<(Settings, Vec<String>), IoError> {
    let bytes = match std::fs::read(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Settings::default(), vec![]));
        }
        other => other?,
    };
    let mut reader = Reader::default();
    match reader.document(&bytes) {
        Ok(settings) => Ok((settings, reader.warnings)),
        Err(cause) => {
            let kept = set_aside(path)?;
            let error = IoError::Settings(format!("kept as {}: {cause}", kept.display()));
            Ok((
                Settings::default(),
                vec![format!("{}: {error}", error.code())],
            ))
        }
    }
}

/// Writes `s` to `path` as `ftgsxr-settings` v1 (§1.11, D-23): to the temporary sibling
/// `<file name>.tmp`, then renamed over `path`, so a crash leaves the old file whole. The
/// directory is created when missing (Q-51). E-IO when it cannot be written.
// Q-49, Q-51
pub fn save_settings(path: &Path, s: &Settings) -> Result<(), IoError> {
    let text = toml::to_string(&document(s)).map_err(std::io::Error::other)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    write_atomic(path, text.as_bytes())
}

/// What a key's value must be, and how it converts: the warning's wording and the check.
type Check<T> = (String, Box<dyn Fn(&Value) -> Option<T>>);

/// A number of either TOML type.
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) => Some(*n as f64),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

/// `true` or `false`.
fn flag() -> Check<bool> {
    ("true or false".into(), Box::new(Value::as_bool))
}

/// A finite number in \[lo, hi\], as f32.
fn range(lo: f64, hi: f64) -> Check<f32> {
    (
        format!("a number in [{lo}, {hi}]"),
        Box::new(move |v| {
            number(v)
                .filter(|x| (lo..=hi).contains(x))
                .map(|x| x as f32)
        }),
    )
}

/// A finite number in E9's \[−4, 4\] other than 0: the clock's speed, which §1.6's
/// presets do not limit (Q-50).
// Q-50
fn speed() -> Check<f64> {
    (
        "a number in [-4, 4] other than 0".into(),
        Box::new(|v| number(v).filter(|x| (-4.0..=4.0).contains(x) && *x != 0.0)),
    )
}

/// One of `values`, as f64.
fn one_of_numbers(values: &'static [f64]) -> Check<f64> {
    let list: Vec<String> = values.iter().map(f64::to_string).collect();
    (
        format!("one of {}", list.join(", ")),
        Box::new(move |v| number(v).filter(|x| values.contains(x))),
    )
}

/// One of the strings `values`.
fn choice(values: &'static [&'static str]) -> Check<String> {
    (
        format!("one of {}", values.join(", ")),
        Box::new(move |v| {
            v.as_str()
                .filter(|s| values.contains(s))
                .map(str::to_string)
        }),
    )
}

/// `N` finite numbers in \[lo, hi\].
fn numbers<const N: usize>(lo: f64, hi: f64) -> Check<[f32; N]> {
    (
        format!("{N} numbers in [{lo}, {}]", upper(hi)),
        Box::new(move |v| {
            let items = v.as_array().filter(|a| a.len() == N)?;
            let mut out = [0.0; N];
            for (o, item) in out.iter_mut().zip(items) {
                *o = number(item).filter(|x| (lo..=hi).contains(x))? as f32;
            }
            Some(out)
        }),
    )
}

/// An integer of at least `min`, as u64.
fn integer(min: u64) -> Check<u64> {
    (
        format!("an integer of at least {min}"),
        Box::new(move |v| {
            v.as_integer()
                .and_then(|n| u64::try_from(n).ok())
                .filter(|n| *n >= min)
        }),
    )
}

/// A non-empty list of distinct reference spaces of D-24.
fn spaces() -> Check<Vec<String>> {
    (
        format!("a list of distinct names from {}", SPACES.join(", ")),
        Box::new(|v| {
            let items = v.as_array().filter(|a| !a.is_empty())?;
            let mut out: Vec<String> = Vec::with_capacity(items.len());
            for item in items {
                let name = item.as_str().filter(|s| SPACES.contains(s))?;
                if out.iter().any(|o| o == name) {
                    return None;
                }
                out.push(name.to_string());
            }
            Some(out)
        }),
    )
}

/// `prefix.key`, or `key` at the top level.
fn dotted(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// The lenient reader: it collects the warnings while it walks the document (§1.11).
#[derive(Default)]
struct Reader {
    warnings: Vec<String>,
}

impl Reader {
    /// The value of `key` in `table` that passes `check`; `None`, with a warning, when
    /// the value does not pass, and `None` silently when the key is absent.
    fn take<T>(
        &mut self,
        table: Option<&Table>,
        prefix: &str,
        key: &str,
        check: Check<T>,
    ) -> Option<T> {
        let value = table?.get(key)?;
        let (expected, convert) = check;
        let out = convert(value);
        if out.is_none() {
            self.wrong(&dotted(prefix, key), value, &expected);
        }
        out
    }

    /// Warns that `path` holds a value that is not `expected` and takes its default.
    fn wrong(&mut self, path: &str, value: &Value, expected: &str) {
        self.warnings.push(format!(
            "settings key `{path}` ({}) is not {expected}; its default is used",
            value.type_str()
        ));
    }

    /// The sub-table `key` of `table`; `None`, with a warning, when it is not a table.
    fn table<'t>(
        &mut self,
        table: Option<&'t Table>,
        prefix: &str,
        key: &str,
    ) -> Option<&'t Table> {
        let value = table?.get(key)?;
        let out = value.as_table();
        if out.is_none() {
            self.wrong(&dotted(prefix, key), value, "a table");
        }
        out
    }

    /// Warns about every key of `table` outside `known` (§1.11 versioning rule).
    fn unknown(&mut self, table: Option<&Table>, prefix: &str, known: &[&str]) {
        for key in table.into_iter().flat_map(Table::keys) {
            if !known.contains(&key.as_str()) {
                self.warnings.push(format!(
                    "unknown settings key `{}` ignored",
                    dotted(prefix, key)
                ));
            }
        }
    }

    /// The whole document, or the reason to set the file aside (Q-51).
    fn document(&mut self, bytes: &[u8]) -> Result<Settings, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8 ({e})"))?;
        let root: Table = text
            .parse()
            .map_err(|e: toml::de::Error| format!("not TOML ({})", e.message()))?;
        match root.get("format") {
            None => self.warnings.push(format!(
                "settings key `format` is missing; read as {FORMAT}"
            )),
            Some(Value::String(f)) if f == FORMAT => {}
            Some(other) => return Err(format!("format {} is not {FORMAT}", shown(other))),
        }
        match root.get("version") {
            None => self.warnings.push(format!(
                "settings key `version` is missing; read as version {VERSION}"
            )),
            Some(v) => match major(v) {
                Some(VERSION) => {}
                Some(m) => return Err(format!("major version {m} is not supported")),
                None => return Err(format!("version {} cannot be read", shown(v))),
            },
        }
        let root = Some(&root);
        self.unknown(root, "", TOP);
        let (display, background) = self.display(root);
        Ok(Settings {
            general: self.general(root),
            sources: self.sources(root),
            playback: self.playback(root),
            display,
            environment: self.environment(root, background),
            locomotion: self.locomotion(root),
            audio: self.audio(root),
            desktop: self.desktop(root),
            xr: self.xr(root),
            cache: self.cache(root),
        })
    }

    fn general(&mut self, root: Option<&Table>) -> GeneralSettings {
        let p = "general";
        let t = self.table(root, "", p);
        self.unknown(t, p, &["tier", "governor", "ui_scale", "log_level"]);
        let d = GeneralSettings::default();
        GeneralSettings {
            tier: self.take(t, p, "tier", choice(TIERS)).unwrap_or(d.tier),
            governor: self.take(t, p, "governor", flag()).unwrap_or(d.governor),
            ui_scale: self
                .take(t, p, "ui_scale", range(0.8, 1.5))
                .unwrap_or(d.ui_scale),
            log_level: self
                .take(t, p, "log_level", choice(LOG_LEVELS))
                .unwrap_or(d.log_level),
        }
    }

    /// `[[sources]]`: an entry without a known `kind` or a string `location` is dropped
    /// with a warning (Q-49).
    fn sources(&mut self, root: Option<&Table>) -> Vec<SourceSpec> {
        let Some(value) = root.and_then(|r| r.get("sources")) else {
            return vec![];
        };
        let Some(items) = value.as_array() else {
            self.wrong("sources", value, "an array of tables");
            return vec![];
        };
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let p = format!("sources[{i}]");
            let Some(t) = item.as_table() else {
                self.wrong(&p, item, "a table; the entry is dropped");
                continue;
            };
            let t = Some(t);
            self.unknown(t, &p, &["kind", "location", "name", "flavor"]);
            let kind = self.take(t, &p, "kind", choice(SOURCE_KINDS));
            let location = self.take(t, &p, "location", text());
            let (Some(kind), Some(location)) = (kind, location) else {
                self.warnings.push(format!(
                    "settings entry `{p}` has no usable kind and location; it is dropped"
                ));
                continue;
            };
            let name = self
                .take(t, &p, "name", text())
                .unwrap_or_else(|| location.clone());
            let flavor = self
                .take(t, &p, "flavor", choice(FLAVORS))
                .and_then(|f| flavor_of(&f));
            out.push(SourceSpec {
                kind,
                location,
                name,
                flavor,
            });
        }
        out
    }

    /// `[playback]`: `loop` read as a `LoopMode` (Q-49, Q-50).
    fn playback(&mut self, root: Option<&Table>) -> PlaybackSettings {
        let p = "playback";
        let t = self.table(root, "", p);
        self.unknown(t, p, &["speed", "loop"]);
        let d = PlaybackSettings::default();
        PlaybackSettings {
            speed: self.take(t, p, "speed", speed()).unwrap_or(d.speed),
            loop_mode: self
                .take(t, p, "loop", choice(LOOP_MODES))
                .and_then(|m| loop_mode_of(&m))
                .unwrap_or(d.loop_mode),
        }
    }

    /// `[display]`, and its `background`, which is the environment's (Q-49).
    fn display(&mut self, root: Option<&Table>) -> (DisplaySettings, [f32; 3]) {
        let p = "display";
        let t = self.table(root, "", p);
        self.unknown(
            t,
            p,
            &[
                "mode",
                "centers_radius",
                "colormap",
                "background",
                "grid",
                "overlays",
            ],
        );
        let d = DisplaySettings::default();
        let background = self
            .take(t, p, "background", numbers::<3>(0.0, 1.0))
            .unwrap_or(EnvironmentSettings::default().background);
        let po = "display.overlays";
        let o = self.table(t, p, "overlays");
        self.unknown(
            o,
            po,
            &["boxes", "velocities", "velocity_every", "time_tint"],
        );
        let od = d.overlays;
        let overlays = OverlaySettings {
            boxes: self.take(o, po, "boxes", flag()).unwrap_or(od.boxes),
            velocities: self
                .take(o, po, "velocities", flag())
                .unwrap_or(od.velocities),
            velocity_every: self
                .take(o, po, "velocity_every", integer(1))
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(od.velocity_every),
            time_tint: self
                .take(o, po, "time_tint", flag())
                .unwrap_or(od.time_tint),
        };
        let display = DisplaySettings {
            mode: self
                .take(t, p, "mode", choice(RENDER_MODES))
                .unwrap_or(d.mode),
            centers_radius: self
                .take(t, p, "centers_radius", range(1.0, 8.0))
                .unwrap_or(d.centers_radius),
            colormap: self
                .take(t, p, "colormap", choice(COLORMAPS))
                .unwrap_or(d.colormap),
            grid: self.take(t, p, "grid", flag()).unwrap_or(d.grid),
            overlays,
        };
        (display, background)
    }

    fn environment(&mut self, root: Option<&Table>, background: [f32; 3]) -> EnvironmentSettings {
        let p = "environment";
        let t = self.table(root, "", p);
        self.unknown(
            t,
            p,
            &[
                "mode",
                "passthrough_brightness",
                "passthrough_contrast",
                "passthrough_saturation",
                "occlusion",
                "occlusion_soft",
                "occlusion_delta_m",
                "hand_removal",
                "lighting_mode",
                "lighting_manual",
                "lighting_gamma",
                "lighting_reference",
                "contact_shadows",
            ],
        );
        let d = EnvironmentSettings::default();
        let pm = "environment.lighting_manual";
        let m = self.table(t, p, "lighting_manual");
        self.unknown(m, pm, &["exposure_ev", "temperature", "tint"]);
        let md = d.lighting_manual;
        let lighting_manual = ManualLighting {
            exposure_ev: self
                .take(m, pm, "exposure_ev", range(-4.0, 4.0))
                .unwrap_or(md.exposure_ev),
            temperature: self
                .take(m, pm, "temperature", range(-1.0, 1.0))
                .unwrap_or(md.temperature),
            tint: self
                .take(m, pm, "tint", range(-1.0, 1.0))
                .unwrap_or(md.tint),
        };
        EnvironmentSettings {
            mode: self
                .take(t, p, "mode", choice(ENVIRONMENT_MODES))
                .unwrap_or(d.mode),
            background,
            passthrough_brightness: self
                .take(t, p, "passthrough_brightness", range(-1.0, 1.0))
                .unwrap_or(d.passthrough_brightness),
            passthrough_contrast: self
                .take(t, p, "passthrough_contrast", range(-1.0, 1.0))
                .unwrap_or(d.passthrough_contrast),
            passthrough_saturation: self
                .take(t, p, "passthrough_saturation", range(-1.0, 1.0))
                .unwrap_or(d.passthrough_saturation),
            occlusion: self.take(t, p, "occlusion", flag()).unwrap_or(d.occlusion),
            occlusion_soft: self
                .take(t, p, "occlusion_soft", flag())
                .unwrap_or(d.occlusion_soft),
            occlusion_delta_m: self
                .take(t, p, "occlusion_delta_m", range(0.0, 0.2))
                .unwrap_or(d.occlusion_delta_m),
            hand_removal: self
                .take(t, p, "hand_removal", flag())
                .unwrap_or(d.hand_removal),
            lighting_mode: self
                .take(t, p, "lighting_mode", choice(LIGHTING_MODES))
                .unwrap_or(d.lighting_mode),
            lighting_manual,
            lighting_gamma: self
                .take(t, p, "lighting_gamma", range(0.0, 1.0))
                .unwrap_or(d.lighting_gamma),
            lighting_reference: self.take(
                t,
                p,
                "lighting_reference",
                numbers::<4>(0.0, f64::from(f32::MAX)),
            ),
            contact_shadows: self
                .take(t, p, "contact_shadows", flag())
                .unwrap_or(d.contact_shadows),
        }
    }

    fn locomotion(&mut self, root: Option<&Table>) -> LocomotionSettings {
        let p = "locomotion";
        let t = self.table(root, "", p);
        self.unknown(
            t,
            p,
            &[
                "speed_mps",
                "snap_turn",
                "snap_deg",
                "smooth_dps",
                "vignette",
                "teleport",
                "height_offset_m",
                "left_handed",
                "reduced_motion",
            ],
        );
        let d = LocomotionSettings::default();
        LocomotionSettings {
            speed_mps: self
                .take(t, p, "speed_mps", range(1.0, 4.0))
                .unwrap_or(d.speed_mps),
            snap_turn: self.take(t, p, "snap_turn", flag()).unwrap_or(d.snap_turn),
            snap_deg: self
                .take(t, p, "snap_deg", one_of_numbers(SNAP_ANGLES))
                .map(|x| x as f32)
                .unwrap_or(d.snap_deg),
            smooth_dps: self
                .take(t, p, "smooth_dps", positive(360.0))
                .unwrap_or(d.smooth_dps),
            vignette: self.take(t, p, "vignette", flag()).unwrap_or(d.vignette),
            teleport: self.take(t, p, "teleport", flag()).unwrap_or(d.teleport),
            height_offset_m: self
                .take(t, p, "height_offset_m", range(-0.5, 0.5))
                .unwrap_or(d.height_offset_m),
            left_handed: self
                .take(t, p, "left_handed", flag())
                .unwrap_or(d.left_handed),
            reduced_motion: self
                .take(t, p, "reduced_motion", flag())
                .unwrap_or(d.reduced_motion),
        }
    }

    fn audio(&mut self, root: Option<&Table>) -> AudioSettings {
        let p = "audio";
        let t = self.table(root, "", p);
        self.unknown(
            t,
            p,
            &["muted", "master", "ui", "soundtrack", "spatial", "sounds"],
        );
        let d = AudioSettings::default();
        let ps = "audio.sounds";
        let s = self.table(t, p, "sounds");
        self.unknown(
            s,
            ps,
            &[
                "hover",
                "select",
                "grab",
                "snap_turn",
                "click",
                "error",
                "confirm",
            ],
        );
        let sd = d.sounds;
        let sounds = SoundSwitches {
            hover: self.take(s, ps, "hover", flag()).unwrap_or(sd.hover),
            select: self.take(s, ps, "select", flag()).unwrap_or(sd.select),
            grab: self.take(s, ps, "grab", flag()).unwrap_or(sd.grab),
            snap_turn: self
                .take(s, ps, "snap_turn", flag())
                .unwrap_or(sd.snap_turn),
            click: self.take(s, ps, "click", flag()).unwrap_or(sd.click),
            error: self.take(s, ps, "error", flag()).unwrap_or(sd.error),
            confirm: self.take(s, ps, "confirm", flag()).unwrap_or(sd.confirm),
        };
        AudioSettings {
            muted: self.take(t, p, "muted", flag()).unwrap_or(d.muted),
            master: self
                .take(t, p, "master", range(0.0, 100.0))
                .unwrap_or(d.master),
            ui: self.take(t, p, "ui", range(0.0, 100.0)).unwrap_or(d.ui),
            soundtrack: self
                .take(t, p, "soundtrack", range(0.0, 100.0))
                .unwrap_or(d.soundtrack),
            spatial: self.take(t, p, "spatial", flag()).unwrap_or(d.spatial),
            sounds,
        }
    }

    /// `[desktop]`: a `keys` entry that is not a string is dropped with a warning.
    fn desktop(&mut self, root: Option<&Table>) -> DesktopSettings {
        let p = "desktop";
        let t = self.table(root, "", p);
        self.unknown(t, p, &["fov_deg", "fly_speed_mps", "keys"]);
        let d = DesktopSettings::default();
        let pk = "desktop.keys";
        let mut keys = BTreeMap::new();
        for (action, key) in self.table(t, p, "keys").into_iter().flatten() {
            match key.as_str() {
                Some(k) => {
                    keys.insert(action.clone(), k.to_string());
                }
                None => self.wrong(
                    &dotted(pk, action),
                    key,
                    "a key name; the binding is dropped",
                ),
            }
        }
        DesktopSettings {
            fov_deg: self
                .take(t, p, "fov_deg", range(40.0, 110.0))
                .unwrap_or(d.fov_deg),
            fly_speed_mps: self
                .take(t, p, "fly_speed_mps", range(0.1, 20.0))
                .unwrap_or(d.fly_speed_mps),
            keys,
        }
    }

    fn xr(&mut self, root: Option<&Table>) -> XrSettings {
        let p = "xr";
        let t = self.table(root, "", p);
        self.unknown(
            t,
            p,
            &[
                "refresh_rate",
                "foveation",
                "render_scale",
                "reference_space",
            ],
        );
        let d = XrSettings::default();
        XrSettings {
            refresh_rate: self.take(t, p, "refresh_rate", positive(f64::from(f32::MAX))),
            foveation: self.take(t, p, "foveation", choice(FOVEATION)),
            render_scale: self.take(t, p, "render_scale", positive(2.0)),
            reference_space: self
                .take(t, p, "reference_space", spaces())
                .unwrap_or(d.reference_space),
        }
    }

    fn cache(&mut self, root: Option<&Table>) -> CacheSettings {
        let p = "cache";
        let t = self.table(root, "", p);
        self.unknown(t, p, &["max_bytes"]);
        CacheSettings {
            max_bytes: self
                .take(t, p, "max_bytes", integer(0))
                .unwrap_or(CacheSettings::default().max_bytes),
        }
    }
}

/// A string value quoted, any other value by its TOML type (for the E-SETTINGS cause).
fn shown(value: &Value) -> String {
    match value.as_str() {
        Some(s) => format!("{s:?}"),
        None => value.type_str().to_string(),
    }
}

/// A string.
fn text() -> Check<String> {
    (
        "a string".into(),
        Box::new(|v| v.as_str().map(str::to_string)),
    )
}

/// An interval's upper bound as the warning shows it.
fn upper(hi: f64) -> String {
    if hi >= f64::from(f32::MAX) {
        "the float32 maximum".into()
    } else {
        hi.to_string()
    }
}

/// A finite number in (0, hi\], as f32.
fn positive(hi: f64) -> Check<f32> {
    (
        format!("a number in (0, {}]", upper(hi)),
        Box::new(move |v| number(v).filter(|x| *x > 0.0 && *x <= hi).map(|x| x as f32)),
    )
}

/// The loop mode a `loop` spelling names (Q-50).
pub(crate) fn loop_mode_of(name: &str) -> Option<LoopMode> {
    match name {
        "loop" => Some(LoopMode::Loop),
        "ping-pong" => Some(LoopMode::PingPong),
        "once" => Some(LoopMode::Once),
        _ => None,
    }
}

/// The `loop` spelling of a loop mode (Q-50).
pub(crate) fn loop_mode_name(mode: LoopMode) -> &'static str {
    match mode {
        LoopMode::Loop => "loop",
        LoopMode::PingPong => "ping-pong",
        LoopMode::Once => "once",
    }
}

/// The `flavor` spellings of §1.11, in `ListingFlavor`'s order.
const FLAVORS: &[&str] = &["auto", "index", "nginx-json", "html"];

/// The flavor a §1.11 spelling names.
pub(crate) fn flavor_of(name: &str) -> Option<ListingFlavor> {
    match name {
        "auto" => Some(ListingFlavor::Auto),
        "index" => Some(ListingFlavor::Index),
        "nginx-json" => Some(ListingFlavor::NginxJson),
        "html" => Some(ListingFlavor::Html),
        _ => None,
    }
}

/// The §1.11 spelling of `flavor`.
pub(crate) fn flavor_name(flavor: ListingFlavor) -> &'static str {
    match flavor {
        ListingFlavor::Auto => "auto",
        ListingFlavor::Index => "index",
        ListingFlavor::NginxJson => "nginx-json",
        ListingFlavor::Html => "html",
    }
}

/// The major version `version` names: an integer, a float's integer part, or the first
/// part of a `"major[.minor]"` string of decimal parts (§1.11, Q-51).
fn major(version: &Value) -> Option<i64> {
    fn decimal(part: &str) -> Option<i64> {
        (!part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            .then(|| part.parse().ok())
            .flatten()
    }
    match version {
        Value::Integer(n) => Some(*n),
        Value::Float(x) if x.is_finite() && *x >= 0.0 && *x < i64::MAX as f64 => {
            Some(x.trunc() as i64)
        }
        Value::String(s) => {
            let mut parts = s.splitn(2, '.');
            let major = parts.next().and_then(decimal)?;
            parts
                .next()
                .is_none_or(|m| decimal(m).is_some())
                .then_some(major)
        }
        _ => None,
    }
}

/// An f32 as the TOML float its shortest decimal form names, so the file reads `0.1`
/// rather than `0.10000000149011612` and reads back to the same f32.
fn float(x: f32) -> Value {
    Value::Float(x.to_string().parse().unwrap_or(f64::from(x)))
}

/// `values` as a TOML array of floats.
fn floats(values: &[f32]) -> Value {
    Value::Array(values.iter().copied().map(float).collect())
}

/// Builds a table from `(key, value)` pairs.
fn table<const N: usize>(pairs: [(&str, Value); N]) -> Table {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// The `ftgsxr-settings` v1 document of `s` (§1.11, Q-49).
fn document(s: &Settings) -> Table {
    let g = &s.general;
    let pb = &s.playback;
    let d = &s.display;
    let e = &s.environment;
    let l = &s.locomotion;
    let a = &s.audio;
    let dk = &s.desktop;
    let x = &s.xr;
    let mut environment = table([
        ("mode", Value::String(e.mode.clone())),
        ("passthrough_brightness", float(e.passthrough_brightness)),
        ("passthrough_contrast", float(e.passthrough_contrast)),
        ("passthrough_saturation", float(e.passthrough_saturation)),
        ("occlusion", Value::Boolean(e.occlusion)),
        ("occlusion_soft", Value::Boolean(e.occlusion_soft)),
        ("occlusion_delta_m", float(e.occlusion_delta_m)),
        ("hand_removal", Value::Boolean(e.hand_removal)),
        ("lighting_mode", Value::String(e.lighting_mode.clone())),
        (
            "lighting_manual",
            Value::Table(table([
                ("exposure_ev", float(e.lighting_manual.exposure_ev)),
                ("temperature", float(e.lighting_manual.temperature)),
                ("tint", float(e.lighting_manual.tint)),
            ])),
        ),
        ("lighting_gamma", float(e.lighting_gamma)),
        ("contact_shadows", Value::Boolean(e.contact_shadows)),
    ]);
    if let Some(r) = e.lighting_reference {
        environment.insert("lighting_reference".into(), floats(&r));
    }
    let mut xr = table([(
        "reference_space",
        Value::Array(
            x.reference_space
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    )]);
    if let Some(r) = x.refresh_rate {
        xr.insert("refresh_rate".into(), float(r));
    }
    if let Some(f) = &x.foveation {
        xr.insert("foveation".into(), Value::String(f.clone()));
    }
    if let Some(r) = x.render_scale {
        xr.insert("render_scale".into(), float(r));
    }
    let sources = s
        .sources
        .iter()
        .map(|src| {
            let mut t = table([
                ("kind", Value::String(src.kind.clone())),
                ("location", Value::String(src.location.clone())),
                ("name", Value::String(src.name.clone())),
            ]);
            if let Some(f) = src.flavor {
                t.insert("flavor".into(), Value::String(flavor_name(f).into()));
            }
            Value::Table(t)
        })
        .collect();
    table([
        ("format", Value::String(FORMAT.into())),
        ("version", Value::Integer(VERSION)),
        (
            "general",
            Value::Table(table([
                ("tier", Value::String(g.tier.clone())),
                ("governor", Value::Boolean(g.governor)),
                ("ui_scale", float(g.ui_scale)),
                ("log_level", Value::String(g.log_level.clone())),
            ])),
        ),
        ("sources", Value::Array(sources)),
        (
            "playback",
            Value::Table(table([
                ("speed", Value::Float(pb.speed)),
                ("loop", Value::String(loop_mode_name(pb.loop_mode).into())),
            ])),
        ),
        (
            "display",
            Value::Table(table([
                ("mode", Value::String(d.mode.clone())),
                ("centers_radius", float(d.centers_radius)),
                ("colormap", Value::String(d.colormap.clone())),
                ("background", floats(&e.background)),
                ("grid", Value::Boolean(d.grid)),
                (
                    "overlays",
                    Value::Table(table([
                        ("boxes", Value::Boolean(d.overlays.boxes)),
                        ("velocities", Value::Boolean(d.overlays.velocities)),
                        (
                            "velocity_every",
                            Value::Integer(i64::from(d.overlays.velocity_every)),
                        ),
                        ("time_tint", Value::Boolean(d.overlays.time_tint)),
                    ])),
                ),
            ])),
        ),
        ("environment", Value::Table(environment)),
        (
            "locomotion",
            Value::Table(table([
                ("speed_mps", float(l.speed_mps)),
                ("snap_turn", Value::Boolean(l.snap_turn)),
                ("snap_deg", float(l.snap_deg)),
                ("smooth_dps", float(l.smooth_dps)),
                ("vignette", Value::Boolean(l.vignette)),
                ("teleport", Value::Boolean(l.teleport)),
                ("height_offset_m", float(l.height_offset_m)),
                ("left_handed", Value::Boolean(l.left_handed)),
                ("reduced_motion", Value::Boolean(l.reduced_motion)),
            ])),
        ),
        (
            "audio",
            Value::Table(table([
                ("muted", Value::Boolean(a.muted)),
                ("master", float(a.master)),
                ("ui", float(a.ui)),
                ("soundtrack", float(a.soundtrack)),
                ("spatial", Value::Boolean(a.spatial)),
                (
                    "sounds",
                    Value::Table(table([
                        ("hover", Value::Boolean(a.sounds.hover)),
                        ("select", Value::Boolean(a.sounds.select)),
                        ("grab", Value::Boolean(a.sounds.grab)),
                        ("snap_turn", Value::Boolean(a.sounds.snap_turn)),
                        ("click", Value::Boolean(a.sounds.click)),
                        ("error", Value::Boolean(a.sounds.error)),
                        ("confirm", Value::Boolean(a.sounds.confirm)),
                    ])),
                ),
            ])),
        ),
        (
            "desktop",
            Value::Table(table([
                ("fov_deg", float(dk.fov_deg)),
                ("fly_speed_mps", float(dk.fly_speed_mps)),
                (
                    "keys",
                    Value::Table(
                        dk.keys
                            .iter()
                            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                            .collect(),
                    ),
                ),
            ])),
        ),
        ("xr", Value::Table(xr)),
        (
            "cache",
            Value::Table(table([(
                "max_bytes",
                Value::Integer(i64::try_from(s.cache.max_bytes).unwrap_or(i64::MAX)),
            )])),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `float`'s shortest decimal, read as f64 then rounded to f32, is the f32 it came
    /// from, over a stride through every finite f32 bit pattern.
    #[test]
    fn float_reads_back_to_the_same_f32() {
        for bits in (0..=u32::MAX).step_by(9_973) {
            let x = f32::from_bits(bits);
            if !x.is_finite() {
                continue;
            }
            let Value::Float(y) = float(x) else {
                panic!("not a float");
            };
            assert_eq!((y as f32).to_bits(), x.to_bits(), "{x:e}");
        }
    }

    /// `major` reads the forms of Q-51 and nothing else.
    #[test]
    fn major_reads_integers_floats_and_dotted_strings() {
        assert_eq!(major(&Value::Integer(2)), Some(2));
        assert_eq!(major(&Value::Float(1.7)), Some(1));
        assert_eq!(major(&Value::String("1.12".into())), Some(1));
        assert_eq!(major(&Value::String("3".into())), Some(3));
        assert_eq!(major(&Value::String("1.x".into())), None);
        assert_eq!(major(&Value::String("".into())), None);
        assert_eq!(major(&Value::Boolean(true)), None);
    }
}
