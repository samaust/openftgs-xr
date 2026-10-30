// SPDX-License-Identifier: Apache-2.0

//! The I/O crate's part of §1.15's error catalog. Defined here so that every module can
//! return it; its public path is the crate-root re-export (§2.1 lists it with `sources`).

use ftgsxr_core::CoreError;

/// Every error of §1.15 the I/O crate can raise, with its code and user text (§1.15,
/// §2.1 `ftgsxr_io`).
///
/// The `String` payloads carry the detail the user text names: the URL and status, the
/// file and what is wrong with it.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// E-NET-FETCH: a listing or a download fails (DNS, TLS, HTTP status, timeout); the
    /// payload gives the URL and the status or cause.
    #[error("Could not reach {0}")]
    Fetch(String),
    /// E-NET-LISTING: no listing format recognized; the payload is the base URL.
    #[error("No index found at {0}")]
    Listing(String),
    /// E-NET-CHECKSUM: a downloaded file's digest disagrees with the index entry,
    /// `SHA256SUMS` or the `.sha256` file; the payload names the file.
    #[error("Download checksum mismatch ({0})")]
    Checksum(String),
    /// E-SETTINGS: a settings file of an unknown major version or an unreadable one; the
    /// payload names the file and the cause.
    #[error("Settings reset, old file kept ({0})")]
    Settings(String),
    /// E-SCENE-FORMAT: a scene, pose or index file breaks its schema; the payload names
    /// the file and the detail.
    #[error("Not a valid scene, pose or index file: {0}")]
    Scene(String),
    /// An error of the core (a model file's reader rules, or for a player file E-JSON when
    /// it is not well-formed JSON and E-FILE-VERSION when its major version is not 1); it
    /// keeps the core's code and user text.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// E-IO: a file cannot be read or written (permissions, a missing directory, a full
    /// disk).
    #[error("Could not read or write a file: {0}")]
    Io(#[from] std::io::Error),
}

impl IoError {
    /// The error's §1.15 code; `Core(e)` has the inner error's code.
    pub fn code(&self) -> &'static str {
        match self {
            IoError::Fetch(_) => "E-NET-FETCH",
            IoError::Listing(_) => "E-NET-LISTING",
            IoError::Checksum(_) => "E-NET-CHECKSUM",
            IoError::Settings(_) => "E-SETTINGS",
            IoError::Scene(_) => "E-SCENE-FORMAT",
            IoError::Core(e) => e.code(),
            IoError::Io(_) => "E-IO",
        }
    }

    /// The text shown to the user as a toast or dialog (§1.15's user-text column);
    /// `Core(e)` has the inner error's text.
    pub fn user_text(&self) -> String {
        match self {
            IoError::Core(e) => e.user_text(),
            other => other.to_string(),
        }
    }
}
