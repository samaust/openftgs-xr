// SPDX-License-Identifier: Apache-2.0

//! R-16's guard in a process that made no Vulkan instance (Q-105, Q-429, #726): a
//! test that only reads a file, as the `ds` lane's `ds_kitchen_1m` and `ds_kitchen_boxes`
//! do, binds the guard (the scan of `layer_guard`), and dropping it passes in a debug
//! build, since there was no instance for the layer to report itself active on. The other
//! side, an instance made without the layer still failing the guard, is
//! `ftgsxr-render`'s `gpu_guard_fails_when_the_validation_layer_is_not_loaded`.
//!
//! The testkit's test binary makes no instance outside its `gpu_` tests, and this test
//! runs in the `cpu` lane, where those don't run.

use ftgsxr_testkit::layer_messages;

#[test]
fn the_guard_passes_a_test_that_made_no_instance() {
    let layer = layer_messages();
    assert_eq!(
        layer.take_active(),
        0,
        "no instance, no layer status message"
    );
    let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(layer)));
    if let Err(panic) = dropped {
        let text = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        panic!("the guard failed a test that made no Vulkan instance: {text}");
    }
}
