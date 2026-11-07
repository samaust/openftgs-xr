// SPDX-License-Identifier: Apache-2.0

//! Dv-9 on the device (§3.4, §1.9 Floor, shadows and anchors, D-26, Q-346). The clauses
//! need `ftgsxr_xr::anchors`' crate-private store, so they live in that crate, inert
//! until an entity named `ftgsxr-test: dv9` exists; this runner spawns it and, on the
//! second launch, an entity named `ftgsxr-anchor: <detail>` with the first launch's
//! verdict detail from the intent extra `ftgsxr-anchor`. The XR crate prints the verdict
//! line (Q-308).

use bevy::prelude::*;

use crate::jni_calls;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv9";
/// The entity name `ftgsxr_xr`'s Dv-9 waits for (Q-308).
// Q-308
const TRIGGER: &str = "ftgsxr-test: dv9";
/// The extra with the first launch's detail, and the entity name's prefix that carries
/// it (Q-346).
// Q-346
const ANCHOR_EXTRA: &str = "ftgsxr-anchor";
const PAYLOAD: &str = "ftgsxr-anchor: ";

/// Adds the test.
pub(super) fn add(app: &mut App) {
    if let Some(detail) = jni_calls::intent_extra(ANCHOR_EXTRA).filter(|d| !d.is_empty()) {
        app.world_mut()
            .spawn(Name::new(format!("{PAYLOAD}{detail}")));
    }
    app.world_mut().spawn(Name::new(TRIGGER));
}
