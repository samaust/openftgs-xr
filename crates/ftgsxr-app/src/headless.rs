// SPDX-License-Identifier: Apache-2.0

//! The headless commands (§1.7, §2.1 `ftgsxr_app::headless`): `render`, `info`,
//! `validate`, `import`, `export` and `bench` (Q-67), and their errors with §1.15's codes and the exit
//! codes 2 and 3.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ftgsxr_core::{
    CoreError, InfoReply, Model, ModelInfo, ModelKind, Pose, RenderConventions, info_json,
    open_model, read_request, sha256, write_exchange, write_renders,
};
use ftgsxr_render::{HeadlessRenderer, QualityTier, RenderError};

mod bench;
// Q-67
pub use bench::{BenchReport, cmd_bench};
mod json;
mod synthetic;

/// A headless command's failure (§1.7, §1.15). `Input` and `Environment` carry their
/// cause's §1.15 code in the message's leading `E-…: ` prefix (Q-64).
#[derive(Clone, Debug, PartialEq)]
pub enum HeadlessError {
    /// Invalid input: a model or request that fails validation or cannot be read or written.
    Input(String),
    /// A numerical failure: a non-finite render (E-NUMERIC).
    Numeric(String),
    /// The environment lacks what the command needs (no GPU adapter, say).
    Environment(String),
}

/// Every code of §1.15's table, to recognize a message's prefix.
const CODES: &[&str] = &[
    "E-FILE-FORMAT",
    "E-FILE-VERSION",
    "E-FILE-KEYS",
    "E-FILE-SHAPE",
    "E-FILE-VALUE",
    "E-FILE-SINGLE",
    "E-FILE-SH",
    "E-FILE-CHECKSUM",
    "E-NET-CHECKSUM",
    "E-IO",
    "E-JSON",
    "E-SCENE-FORMAT",
    "E-GPU-SHADER",
    "E-REQ-UNIT",
    "E-REQ-DEGREE",
    "E-MEM-BUDGET",
    "E-NET-FETCH",
    "E-NET-LISTING",
    "E-GPU-ADAPTER",
    "E-GPU-LOST",
    "E-XR-RUNTIME",
    "E-XR-LOST",
    "E-XR-PERMISSION",
    "E-XR-UNSUPPORTED",
    "E-AUDIO",
    "E-SCENE-MISSING",
    "E-SETTINGS",
    "E-NUMERIC",
];

/// The §1.15 code at the head of `message` (`<code>: …`) and the rest, if it has one.
fn split_code(message: &str) -> Option<(&'static str, &str)> {
    let (head, rest) = message.split_once(':')?;
    let code = CODES.iter().find(|c| **c == head)?;
    Some((code, rest.trim_start()))
}

impl HeadlessError {
    /// The error's §1.15 code (Q-64): the message's prefix for `Input` and `Environment`,
    /// E-NUMERIC for `Numeric`. A message without a §1.15 prefix falls back to
    /// E-FILE-FORMAT for `Input` and E-GPU-ADAPTER for `Environment` (Q-75).
    // Q-64, Q-75
    pub fn code(&self) -> &'static str {
        match self {
            HeadlessError::Input(m) => split_code(m).map_or("E-FILE-FORMAT", |(c, _)| c),
            HeadlessError::Environment(m) => split_code(m).map_or("E-GPU-ADAPTER", |(c, _)| c),
            HeadlessError::Numeric(_) => "E-NUMERIC",
        }
    }

    /// The text shown to the user (§1.15's user-text column, Q-64): the message without its
    /// code prefix, or for `Numeric` §1.15's "Render produced invalid values" with the
    /// detail in parentheses.
    // Q-64
    pub fn user_text(&self) -> String {
        match self {
            HeadlessError::Input(m) | HeadlessError::Environment(m) => split_code(m)
                .map_or(m.as_str(), |(_, rest)| rest)
                .to_owned(),
            HeadlessError::Numeric(d) if d.is_empty() => {
                "Render produced invalid values".to_owned()
            }
            HeadlessError::Numeric(d) => format!("Render produced invalid values ({d})"),
        }
    }
}

impl fmt::Display for HeadlessError {
    /// `<code>: <user text>`, the line the command line prints.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.user_text())
    }
}

impl std::error::Error for HeadlessError {}

/// The command line's exit code for `e` (§1.7, §1.15): 2 for `Input` and `Environment`,
/// 3 for `Numeric`.
pub fn exit_code(e: &HeadlessError) -> i32 {
    match e {
        HeadlessError::Input(_) | HeadlessError::Environment(_) => 2,
        HeadlessError::Numeric(_) => 3,
    }
}

/// `ftgsxr render MODEL REQUEST.json OUT.safetensors [--tier TIER]` (§1.7): answers a
/// `ftgs-render-request` headless. It reads the request and the model (any §1.2 input),
/// checks the request against the model (E-REQ-UNIT, E-REQ-DEGREE; exit 2) before it opens
/// the GPU, renders each item through `HeadlessRenderer` in `tier` at its timestamp from
/// its camera, at the identity pose, with the request's background and SH degree (`None`
/// is the model's L), and writes `out` as `ftgs-renders` with `producer` = `openftgs-xr
/// <version> (<GPU adapter name>)` and `request_sha256` = the SHA-256 of the request
/// file's bytes. The file is written once every item has rendered, so a failure, a
/// non-finite render among them (E-NUMERIC, exit 3), writes no file (Q-90).
// Q-90, Q-86
pub fn cmd_render(
    model: &Path,
    request: &Path,
    out: &Path,
    tier: QualityTier,
    version: &str,
) -> Result<(), HeadlessError> {
    let bytes = fs::read(request).map_err(|e| input_error(request, e))?;
    let req = read_request(&bytes).map_err(|e| core_input(request, &e))?;
    let loaded = load(model)?;
    if req.time_unit != loaded.time.time_unit {
        return Err(core_input(
            request,
            &CoreError::RequestUnit {
                request: req.time_unit,
                model: loaded.time.time_unit,
            },
        ));
    }
    if let Some(degree) = req.sh_degree.filter(|&d| d > loaded.sh_degree) {
        // Q-64: no `CoreError` carries E-REQ-DEGREE, so the code is the message's prefix.
        return Err(HeadlessError::Input(format!(
            "E-REQ-DEGREE: {}: Request degree {degree}, model degree {}",
            request.display(),
            loaded.sh_degree
        )));
    }
    let mut renderer = HeadlessRenderer::new(tier).map_err(|e| render_error(model, e))?;
    let id = renderer
        .load(&loaded, req.sh_degree)
        .map_err(|e| render_error(model, e))?;
    let renders = req
        .items
        .iter()
        .map(|item| {
            renderer
                .render(
                    &[(id, Pose::IDENTITY)],
                    &item.camera,
                    item.timestamp,
                    req.background,
                )
                .map(|image| (item.id.clone(), image))
                .map_err(|e| match e {
                    RenderError::NonFinite => HeadlessError::Numeric(format!("item {}", item.id)),
                    other => render_error(model, other),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let producer = format!("openftgs-xr {version} ({})", renderer.adapter_name());
    write_renders(out, &renders, &producer, req.sha256).map_err(|e| core_input(out, &e))
}

/// A render-layer failure as a `HeadlessError` (Q-90): a core error on the model as
/// `core_input`, a non-finite render as `Numeric`, every GPU condition as `Environment`
/// with its §1.15 code (all exit 2, §1.15).
// Q-90, Q-64
fn render_error(model: &Path, e: RenderError) -> HeadlessError {
    match e {
        RenderError::Core(core) => core_input(model, &core),
        RenderError::NonFinite => HeadlessError::Numeric(String::new()),
        other => HeadlessError::Environment(format!("{}: {}", other.code(), other.user_text())),
    }
}

/// `ftgsxr info [--json]` (§1.7): the adapter reply of openftgs §3.3's contract, with name
/// `openftgs-xr`, the given version, no query kinds, the rasterizer `openftgs-xr
/// wgpu-blend` at the same version, and openftgs's conventions with `transmittance_stop`
/// null (D-7). With `json`, `info_json`'s compact object; without it, one `key: value`
/// line per field in the same order, `rasterizer` as `<name> <version>` and each
/// convention as `render_conventions.<key>: <value>` (Q-71), strings bare and other values
/// as their JSON text (Q-77). No trailing newline.
// Q-71, Q-77
pub fn cmd_info(version: &str, json: bool) -> String {
    let reply = InfoReply {
        name: "openftgs-xr".to_owned(),
        version: version.to_owned(),
        query_kinds: Vec::new(),
        rasterizer_name: "openftgs-xr wgpu-blend".to_owned(),
        rasterizer_version: version.to_owned(),
        // D-7: E7 applies no transmittance stop.
        render_conventions: RenderConventions {
            transmittance_stop: None,
            ..RenderConventions::default()
        },
    };
    if json {
        return info_json(&reply);
    }
    let kinds = reply
        .query_kinds
        .iter()
        .map(|k| format!("{k:?}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut lines = vec![
        format!("name: {}", reply.name),
        format!("version: {}", reply.version),
        format!("query_kinds: [{kinds}]"),
        format!(
            "rasterizer: {} {}",
            reply.rasterizer_name, reply.rasterizer_version
        ),
    ];
    for (key, value) in json_entries(&reply.render_conventions.to_json()) {
        lines.push(format!("render_conventions.{key}: {value}"));
    }
    lines.join("\n")
}

/// The entries of a flat compact JSON object as `RenderConventions::to_json` writes it, in
/// its order: each key, and its value's JSON text with a string's quotes removed. The
/// object's values are numbers, `null`, strings without commas or escapes, and one array
/// of numbers, so an entry ends at the next `,"`.
fn json_entries(object: &str) -> Vec<(String, String)> {
    let inner = object
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(object);
    inner
        .split(",\"")
        .filter_map(|entry| {
            let (key, value) = entry.split_once("\":")?;
            let key = key.trim_start_matches('"');
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(value);
            Some((key.to_owned(), value.to_owned()))
        })
        .collect()
}

/// `Input("<code>: <path>: <user text>")` for a core error met on `path` (Q-75).
// Q-75
fn core_input(path: &Path, e: &CoreError) -> HeadlessError {
    HeadlessError::Input(format!(
        "{}: {}: {}",
        e.code(),
        path.display(),
        e.user_text()
    ))
}

/// Reads and validates the model at `path` (§1.2): `open_model`, its errors as `core_input`.
fn load(path: &Path) -> Result<Model, HeadlessError> {
    open_model(path).map_err(|e| core_input(path, &e))
}

/// The size of the file a model's identity hashes (D-40): `params.safetensors` for a
/// checkpoint directory, the file itself otherwise (Q-76).
// Q-76
fn identity_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("params.safetensors")
    } else {
        path.to_path_buf()
    }
}

/// `ftgsxr validate MODEL ...` (§1.7): reads and validates each model, in order, and
/// returns its header information: N, L, the time span, the producer, the conventions,
/// the kind and the size of the file its identity hashes (Q-76). The first invalid
/// file stops the run with its E-FILE code (exit 2).
pub fn cmd_validate(models: &[PathBuf]) -> Result<Vec<ModelInfo>, HeadlessError> {
    models
        .iter()
        .map(|path| {
            let model = load(path)?;
            let file = identity_file(path);
            let file_bytes = fs::metadata(&file)
                .map_err(|e| core_input(&file, &CoreError::Io(e)))?
                .len();
            Ok(ModelInfo {
                n: model.len() as u64,
                sh_degree: model.sh_degree,
                time: model.time,
                producer: model.producer,
                conventions: model.conventions,
                kind: model.kind,
                file_bytes,
            })
        })
        .collect()
}

/// `ftgsxr import IF OUT` (§1.7): as `cmd_export`; §1.7 gives the two commands one rule.
pub fn cmd_import(input: &Path, out: &Path) -> Result<(), HeadlessError> {
    convert(input, out)
}

/// `ftgsxr export MODEL OUT` (§1.7): reads and validates `model` (§1.2), then writes a
/// `ftgs-if` file at `out`: a byte copy of a `ftgs-if` input, and the activated exchange
/// form of a checkpoint or a PLY (`write_exchange`, producer `openftgs-xr <version>`;
/// Q-65, Q-72). An invalid input writes nothing (exit 2).
pub fn cmd_export(model: &Path, out: &Path) -> Result<(), HeadlessError> {
    convert(model, out)
}

/// The rule `import` and `export` share (§1.7).
// Q-65, Q-72
fn convert(input: &Path, out: &Path) -> Result<(), HeadlessError> {
    let model = load(input)?;
    match model.kind {
        ModelKind::ExchangeFile => {
            let bytes = fs::read(input).map_err(|e| input_error(input, e))?;
            // The bytes copied are the bytes validated: a file replaced between the two
            // reads is refused rather than copied unchecked.
            if sha256(&bytes) != model.identity {
                return Err(core_input(
                    input,
                    &CoreError::Checksum(format!("{} changed while read", input.display())),
                ));
            }
            write_atomic(out, &bytes).map_err(|e| input_error(out, e))
        }
        ModelKind::Checkpoint { .. } | ModelKind::Ply => {
            write_exchange(out, &model).map_err(|e| core_input(out, &e))
        }
    }
}

/// `core_input` for an I/O error on `path` (E-IO).
fn input_error(path: &Path, e: std::io::Error) -> HeadlessError {
    core_input(path, &CoreError::Io(e))
}

/// Writes `bytes` at `path` through a temporary sibling `.<name>.tmp-<pid>` renamed over
/// it, as `write_exchange` writes (openftgs Q-126): a failed write leaves the old file, or
/// none, and no temporary.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} names no file", path.display()),
        )
    })?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    let written = fs::File::create(&temporary)
        .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
        .and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}
