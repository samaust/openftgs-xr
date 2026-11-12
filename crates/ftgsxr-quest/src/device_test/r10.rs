// SPDX-License-Identifier: Apache-2.0

//! R-10 on the device (§3.2 R-10, D-12, Q-122, Q-70, Q-427): the tier bounds on the
//! Quest's GPU through `cmd_render`. The host driver pushed the degree-2 random model
//! `dv/r10-random.safetensors` with its request `dv/r10-random.request.json`, and
//! `dv/R10_deg2.safetensors` with `dv/R10_deg2.request.json`. The random model is rendered
//! in `quest-balanced` and in `conformance` (`dv/r10-random.<tier>.renders.safetensors`),
//! and `R10_deg2` in `conformance` at its request's degree 2 (`dv/R10_deg2.renders
//! .safetensors`); the driver judges them (Q-122).

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use ftgsxr_app::{QualityTier, cmd_render};

/// The test's name in the intent extra (Q-427).
pub(crate) const NAME: &str = "r10";

/// Renders the three files in `dv`: `Ok` once every file is written, or the first failure
/// with its model, tier and code.
pub(super) fn run(dv: &Path) -> Result<String, String> {
    let runs = [
        ("r10-random", "conformance", QualityTier::conformance()),
        (
            "r10-random",
            "quest-balanced",
            QualityTier::quest_balanced(),
        ),
        ("R10_deg2", "conformance", QualityTier::conformance()),
    ];
    let count = runs.len();
    for (model, tier_name, tier) in runs {
        let out = if model == "R10_deg2" {
            dv.join("R10_deg2.renders.safetensors")
        } else {
            dv.join(format!("{model}.{tier_name}.renders.safetensors"))
        };
        // A file of an earlier run must not stand for this one.
        let _ = std::fs::remove_file(&out);
        cmd_render(
            &dv.join(format!("{model}.safetensors")),
            &dv.join(format!("{model}.request.json")),
            &out,
            tier,
            env!("CARGO_PKG_VERSION"),
        )
        .map_err(|e| format!("{model} {tier_name}: {}: {}", e.code(), e.user_text()))?;
        // The app writes its files 0600; the host's `adb pull` runs as the shell user, who
        // needs them readable.
        std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o644))
            .map_err(|e| format!("{model} {tier_name}: {}: {e}", out.display()))?;
    }
    Ok(format!("{count} files rendered"))
}
