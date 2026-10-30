// SPDX-License-Identifier: Apache-2.0

//! The headless commands (§1.7, §2.1 `ftgsxr_app::headless`): `info`, `validate`, `import`
//! and `export`, and their errors with §1.15's codes and the exit codes 2 and 3.

use std::fmt;

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
