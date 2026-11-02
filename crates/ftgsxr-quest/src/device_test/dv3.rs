// SPDX-License-Identifier: Apache-2.0

//! Dv-3 on the device (§3.4, Q-122): R-4's three checks (§3.2) on the two models the host
//! driver pushed into `dv/`, through `ftgsxr_app::sort_check`.

use std::path::Path;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv3";

/// `Ok` with a summary, or the first disagreement.
pub(super) fn run(dv: &Path) -> Result<String, String> {
    // Q-146
    ftgsxr_app::sort_check(
        &dv.join("r4-random.safetensors"),
        &dv.join("r4-repeated.safetensors"),
    )
}
