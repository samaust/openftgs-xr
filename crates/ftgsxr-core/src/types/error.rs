// SPDX-License-Identifier: Apache-2.0

//! The core's part of §1.15's error catalog.

/// Every error of §1.15 the core can raise, with its code and user text.
///
/// The `String` payloads carry the detail the user text names: the offending key, value
/// or shape, or what was found where something else was expected.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// E-FILE-FORMAT: `format` is not the expected one, or the container is not
    /// safetensors, JSON or PLY; the payload says what was found.
    #[error("Not a supported model file ({0})")]
    Format(String),
    /// E-FILE-VERSION: an unknown major version; the payload is the version found.
    #[error("Version {0} is newer than this player supports")]
    Version(String),
    /// E-FILE-KEYS: a required tensor or metadata key is missing; the payload is the key.
    #[error("Missing key {0}")]
    Keys(String),
    /// E-FILE-SHAPE: a dtype or shape disagrees with §1.2; the payload names the key, what
    /// it has and what was expected.
    #[error("Wrong dtype or shape: {0}")]
    Shape(String),
    /// E-FILE-VALUE: a non-finite value, a non-positive duration or scale, an opacity
    /// outside (0, 1), a rotation of norm ≤ 10⁻⁸, τ₁ < τ₀, a background outside \[0, 1\],
    /// a metadata number that does not parse; the payload names the key and the value.
    #[error("{0} has an invalid value")]
    Value(String),
    /// E-FILE-SINGLE: τ₀ = τ₁ with a time center ≠ τ₀ or a velocity ≠ 0; the payload
    /// names the row.
    #[error("Single-timestamp model with moving primitives ({0})")]
    SingleTimestamp(String),
    /// E-FILE-SH: `sh_degree` not in 0–3 or disagreeing with the `sh` shape; the payload
    /// gives the degree and the shape.
    #[error("Unsupported SH degree ({0})")]
    Sh(String),
    /// E-FILE-CHECKSUM: a checkpoint's `files` digest disagrees with `params.safetensors`,
    /// or a renders file's `request_sha256` with the request's digest; the payload names
    /// the file and the two digests.
    #[error("Checksum mismatch ({0})")]
    Checksum(String),
    /// E-REQ-UNIT: a request's `time_unit` differs from the model's.
    #[error("Request time unit {request}, model {model}")]
    RequestUnit {
        /// The request's `time_unit`.
        request: String,
        /// The model's `time_unit`.
        model: String,
    },
    /// E-IO: a file cannot be read or written.
    #[error("Could not read or write a file: {0}")]
    Io(#[from] std::io::Error),
    /// E-JSON: a player file or a request is not well-formed JSON; the payload names the
    /// file and the parser's message.
    #[error("{0} is not valid JSON")]
    Json(String),
}

impl CoreError {
    /// The error's §1.15 code.
    pub fn code(&self) -> &'static str {
        match self {
            CoreError::Format(_) => "E-FILE-FORMAT",
            CoreError::Version(_) => "E-FILE-VERSION",
            CoreError::Keys(_) => "E-FILE-KEYS",
            CoreError::Shape(_) => "E-FILE-SHAPE",
            CoreError::Value(_) => "E-FILE-VALUE",
            CoreError::SingleTimestamp(_) => "E-FILE-SINGLE",
            CoreError::Sh(_) => "E-FILE-SH",
            CoreError::Checksum(_) => "E-FILE-CHECKSUM",
            CoreError::RequestUnit { .. } => "E-REQ-UNIT",
            CoreError::Io(_) => "E-IO",
            CoreError::Json(_) => "E-JSON",
        }
    }

    /// The text shown to the user as a toast or dialog (§1.15's user-text column).
    pub fn user_text(&self) -> String {
        self.to_string()
    }
}
