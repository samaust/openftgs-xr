// SPDX-License-Identifier: Apache-2.0

//! Dv-8 on the device (§3.4, §1.8 Hands and Controllers, D-25, D-44, Appendix D.3, Q-272).
//! The clauses need `ftgsxr_xr::tracking`'s crate-private items (the trackers, the raw
//! joint locates, the haptic output), so they live in that crate, inert until an entity
//! named `ftgsxr-test: dv8` exists; this runner spawns it, and the XR crate prints the
//! verdict line (Q-308).

use bevy::prelude::*;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv8";
/// The entity name `ftgsxr_xr`'s Dv-8 waits for (Q-308).
// Q-308
const TRIGGER: &str = "ftgsxr-test: dv8";

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.world_mut().spawn(Name::new(TRIGGER));
}
