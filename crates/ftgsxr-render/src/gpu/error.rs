// SPDX-License-Identifier: Apache-2.0

//! The render crate's part of §1.15's error catalog (§2.1 `RenderError`, Q-64).

use std::fmt;

use ftgsxr_core::CoreError;

/// Every error of §1.15 the render layer can raise, with its code and user text (§1.15,
/// §2.1 `ftgsxr_render::gpu`; `code` and `user_text` are Q-64's).
#[derive(Debug)]
pub enum RenderError {
    /// E-GPU-ADAPTER: no suitable graphics adapter.
    NoAdapter,
    /// E-GPU-LOST: the graphics device is lost.
    DeviceLost,
    /// E-MEM-BUDGET: the model's static bytes exceed what the device's live models leave of
    /// its budget (§1.2 load option 4, D-13, Q-74); both figures in bytes.
    Budget {
        /// The static GPU bytes the model needs (§1.12).
        needed: u64,
        /// The budget left: the device's budget less its live models' bytes (Q-74).
        budget: u64,
    },
    /// E-GPU-SHADER: a shader fails to compile or a pipeline cannot be created; the
    /// payload is the detail.
    Shader(String),
    /// E-NUMERIC: a non-finite value in a render.
    NonFinite,
    /// An error of the core (a model the render layer reads); it keeps the core's code
    /// and user text.
    Core(CoreError),
}

/// Bytes to decimal megabytes for the user text, rounded up (a need) or down (a budget),
/// so a refused model never reads as fitting.
fn mb(bytes: u64, up: bool) -> u64 {
    if up {
        bytes.div_ceil(1_000_000)
    } else {
        bytes / 1_000_000
    }
}

impl RenderError {
    /// The error's §1.15 code; `Core(e)` has the inner error's code (Q-64).
    pub fn code(&self) -> &'static str {
        // Q-64
        match self {
            RenderError::NoAdapter => "E-GPU-ADAPTER",
            RenderError::DeviceLost => "E-GPU-LOST",
            RenderError::Budget { .. } => "E-MEM-BUDGET",
            RenderError::Shader(_) => "E-GPU-SHADER",
            RenderError::NonFinite => "E-NUMERIC",
            RenderError::Core(e) => e.code(),
        }
    }

    /// The text shown to the user as a toast or dialog (§1.15's user-text column);
    /// `Core(e)` has the inner error's text (Q-64).
    pub fn user_text(&self) -> String {
        match self {
            RenderError::Core(e) => e.user_text(),
            other => other.to_string(),
        }
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::NoAdapter => f.write_str("No Vulkan adapter found"),
            RenderError::DeviceLost => f.write_str("Graphics device lost; reloading"),
            RenderError::Budget { needed, budget } => write!(
                f,
                "Needs {} MB, budget {} MB; reduce to budget or choose a tier",
                mb(*needed, true),
                mb(*budget, false)
            ),
            RenderError::Shader(detail) => write!(f, "Shader compilation failed ({detail})"),
            RenderError::NonFinite => f.write_str("Render produced invalid values"),
            RenderError::Core(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RenderError::Core(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CoreError> for RenderError {
    fn from(e: CoreError) -> Self {
        RenderError::Core(e)
    }
}
