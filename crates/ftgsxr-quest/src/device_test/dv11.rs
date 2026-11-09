// SPDX-License-Identifier: Apache-2.0

//! Dv-11 on the device (§3.4, §1.9 Light estimation, E14, Q-390). The clauses read
//! `ftgsxr_xr::light`'s crate-private control, so
//! they live in that crate, inert until an entity named `ftgsxr-test: dv11` exists; this
//! runner spawns it, and the XR crate prints the verdict line (Q-308).

use bevy::prelude::*;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv11";
/// The entity name `ftgsxr_xr`'s Dv-11 waits for (Q-308).
// Q-308, Q-390
const TRIGGER: &str = "ftgsxr-test: dv11";

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.world_mut().spawn(Name::new(TRIGGER));
}
