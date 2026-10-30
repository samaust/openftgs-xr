// SPDX-License-Identifier: Apache-2.0

//! The headless commands (§1.7, §2.1 `ftgsxr_app::headless`): `info`, `validate`, `import`
//! and `export`, and their errors with §1.15's codes and the exit codes 2 and 3.

use std::fmt;

use ftgsxr_core::{InfoReply, RenderConventions, info_json};

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
    /// E-FILE-FORMAT for `Input` and E-GPU-ADAPTER for `Environment` (Q-T117.1).
    // Q-64, Q-T117.1
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

/// `ftgsxr info [--json]` (§1.7): the adapter reply of openftgs §3.3's contract, with name
/// `openftgs-xr`, the given version, no query kinds, the rasterizer `openftgs-xr
/// wgpu-blend` at the same version, and openftgs's conventions with `transmittance_stop`
/// null (D-7). With `json`, `info_json`'s compact object; without it, one `key: value`
/// line per field in the same order, `rasterizer` as `<name> <version>` and each
/// convention as `render_conventions.<key>: <value>` (Q-71), strings bare and other values
/// as their JSON text (Q-T117.3). No trailing newline.
// Q-71, Q-T117.3
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
