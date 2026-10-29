// SPDX-License-Identifier: Apache-2.0

//! A model's time span (openftgs §1.4 Time; §1.14 Time).

use super::CoreError;

/// τ₀, τ₁ and the unit of a model's raw time (openftgs §1.4 Time).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeSpan {
    /// τ₀, the start of the training span, raw τ.
    pub tau0: f64,
    /// τ₁, the end of the training span, raw τ.
    pub tau1: f64,
    /// The unit of τ (`s` from openftgs's converters).
    pub time_unit: String,
}

impl TimeSpan {
    /// T_τ = τ₁ − τ₀, 0 with a single timestamp.
    pub fn span(&self) -> f64 {
        self.tau1 - self.tau0
    }

    /// Whether τ₀ = τ₁ (D-9).
    pub fn is_single(&self) -> bool {
        self.tau0 == self.tau1
    }

    /// E9's r, the model's time units per second: 1 for `s`, 1000 for `ms`;
    /// `Err(CoreError::Value)` for any other unit, where the caller warns and uses 1 (D-32).
    pub fn rate_per_second(&self) -> Result<f64, CoreError> {
        match self.time_unit.as_str() {
            "s" => Ok(1.0),
            "ms" => Ok(1000.0),
            other => Err(CoreError::Value(format!("time_unit {other:?}"))),
        }
    }
}
