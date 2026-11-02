// SPDX-License-Identifier: Apache-2.0

//! Dv-4 on the device (§3.4, Q-122): the probes P1–P10 rendered by `cmd_render` in the
//! `conformance` tier on the Quest's GPU (§1.7 `render`). The host driver pushed
//! `dv/<P>.safetensors` and `dv/<P>.request.json`; each render is written to
//! `dv/<P>.renders.safetensors`, and the driver judges them with E15's tier (Q-122).

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use ftgsxr_app::{QualityTier, cmd_render};

/// The probes (§3.2 R-1).
const PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

/// Renders every probe in `dv`: `Ok` with the count once every file is written, or the
/// first failure.
pub(super) fn run(dv: &Path) -> Result<String, String> {
    for p in PROBES {
        let out = dv.join(format!("{p}.renders.safetensors"));
        // A file of an earlier run must not stand for this one.
        let _ = std::fs::remove_file(&out);
        cmd_render(
            &dv.join(format!("{p}.safetensors")),
            &dv.join(format!("{p}.request.json")),
            &out,
            QualityTier::conformance(),
            env!("CARGO_PKG_VERSION"),
        )
        .map_err(|e| format!("{p}: {}: {}", e.code(), e.user_text()))?;
        // The app writes its files 0600; the host's `adb pull` runs as the shell user,
        // who needs them readable.
        std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o644))
            .map_err(|e| format!("{p}: {}: {e}", out.display()))?;
    }
    Ok(format!("{} probes rendered", PROBES.len()))
}
