// SPDX-License-Identifier: Apache-2.0

//! Dv-6 on the device (§3.4, §1.9, D-14, Q-346). The clauses read
//! `ftgsxr_xr::passthrough`'s crate-private record of its layer and the render world's
//! layer list, so they live in that crate, inert until an entity named
//! `ftgsxr-test: dv6` exists; this runner spawns it, and the XR crate prints the verdict
//! line (Q-308).

use bevy::prelude::*;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv6";
/// The entity name `ftgsxr_xr`'s Dv-6 waits for (Q-308).
// Q-308, Q-346
const TRIGGER: &str = "ftgsxr-test: dv6";

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.world_mut().spawn(Name::new(TRIGGER));
}
