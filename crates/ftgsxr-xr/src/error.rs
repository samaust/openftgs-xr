// SPDX-License-Identifier: Apache-2.0

//! The XR errors (§2.1 `XrError`) with §1.15's codes and user texts (Q-119).

use std::fmt;

use ftgsxr_app::Permission;

/// An XR failure (§2.1, §1.15).
#[derive(Clone, Debug, PartialEq)]
pub enum XrError {
    /// An extension the feature needs is absent (E-XR-UNSUPPORTED).
    Unsupported(&'static str),
    /// A permission was denied (E-XR-PERMISSION).
    Permission(Permission),
    /// No OpenXR runtime, or the session cannot be created (E-XR-RUNTIME).
    Runtime(String),
    /// The session is lost (E-XR-LOST).
    SessionLost,
}

impl XrError {
    /// The error's §1.15 code.
    // Q-119
    pub fn code(&self) -> &'static str {
        match self {
            XrError::Unsupported(_) => "E-XR-UNSUPPORTED",
            XrError::Permission(_) => "E-XR-PERMISSION",
            XrError::Runtime(_) => "E-XR-RUNTIME",
            XrError::SessionLost => "E-XR-LOST",
        }
    }

    /// The text shown to the user (§1.15's user-text column, the detail in parentheses).
    // Q-119
    pub fn user_text(&self) -> String {
        match self {
            XrError::Unsupported(what) => format!("Not available on this runtime ({what})"),
            XrError::Permission(p) => format!("Permission {p:?} denied; feature unavailable"),
            XrError::Runtime(msg) => format!("No OpenXR runtime ({msg})"),
            XrError::SessionLost => "XR session lost".to_owned(),
        }
    }
}

impl fmt::Display for XrError {
    /// `<code>: <user text>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.user_text())
    }
}

impl std::error::Error for XrError {}
