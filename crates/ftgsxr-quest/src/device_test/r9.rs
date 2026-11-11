// SPDX-License-Identifier: Apache-2.0

//! R-9 on the device (§3.2 R-9, D-4, Q-122, Q-414, Q-T728.2): the probes P1–P10 rendered
//! for R-9's two eyes on the Quest's GPU by `cmd_render_stereo`, per view and
//! shared-center in `conformance`, and once more shared-center with `half_records`
//! (§1.12). The host driver pushed `dv/<P>.safetensors` and `dv/R9_<P>.request.json`; the
//! renders are written to `dv/R9_<P>.<mode>.renders.safetensors`, and the driver judges
//! them with E15 (Q-122).

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use ftgsxr_app::{QualityTier, SortMode, cmd_render_stereo};

/// The test's name in the intent extra (Q-T728.2).
pub(crate) const NAME: &str = "r9";
/// The probes (§3.2 R-1).
const PROBES: [&str; 10] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"];

/// The three runs of each probe: the file's mode name, the tier and the sort mode.
fn modes() -> [(&'static str, QualityTier, SortMode); 3] {
    [
        ("per-view", QualityTier::conformance(), SortMode::PerView),
        ("shared", QualityTier::conformance(), SortMode::SharedCenter),
        (
            "shared-half",
            QualityTier {
                half_records: true,
                ..QualityTier::conformance()
            },
            SortMode::SharedCenter,
        ),
    ]
}

/// Renders every probe in every mode in `dv`: `Ok` with the count once every file is
/// written, or the first failure with its probe, mode and code.
pub(super) fn run(dv: &Path) -> Result<String, String> {
    let mut files = 0;
    for p in PROBES {
        for (mode, tier, sort) in modes() {
            let out = dv.join(format!("R9_{p}.{mode}.renders.safetensors"));
            // A file of an earlier run must not stand for this one.
            let _ = std::fs::remove_file(&out);
            // Q-414
            cmd_render_stereo(
                &dv.join(format!("{p}.safetensors")),
                &dv.join(format!("R9_{p}.request.json")),
                &out,
                tier,
                sort,
                env!("CARGO_PKG_VERSION"),
            )
            .map_err(|e| format!("R9_{p} {mode}: {}: {}", e.code(), e.user_text()))?;
            // The app writes its files 0600; the host's `adb pull` runs as the shell user,
            // who needs them readable.
            std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o644))
                .map_err(|e| format!("R9_{p} {mode}: {}: {e}", out.display()))?;
            files += 1;
        }
    }
    Ok(format!("{files} files rendered"))
}
