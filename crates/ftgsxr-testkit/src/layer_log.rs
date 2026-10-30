// SPDX-License-Identifier: Apache-2.0

//! The Vulkan validation layer's messages for the `gpu` lane (§1.13, R-16, Q-T121.1).
//!
//! wgpu-hal hands the layer's messages (its `VK_EXT_debug_utils` messenger) to the `log`
//! crate under the target `wgpu_hal::vulkan::instance`, never to the device's
//! uncaptured-error handler that `SplatDevice`'s collector holds (Q-74). The messenger's
//! severities are chosen when the instance is created, from `log`'s maximum level then: with
//! no logger only errors, and those are dropped. This module installs, once per process, a
//! `tracing` subscriber with `log`'s bridge (tracing-subscriber's `tracing-log` feature) at
//! the debug level, so the messenger subscribes to every severity, and keeps the layer's
//! messages; a [`LayerMessages`] guard, made before the test's first device, fails the test
//! on any warning or error among them.

use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock, PoisonError};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// The `log` target of wgpu-hal's Vulkan messenger callback.
const MESSENGER_TARGET: &str = "wgpu_hal::vulkan::instance";

/// The Khronos layer's status message at instance creation, an INFO-severity message.
const LAYER_ACTIVE: &str = "Khronos Validation Layer Active";

/// What the subscriber has kept since the last take.
#[derive(Default)]
struct Kept {
    /// The layer's warnings and errors, as `"<LEVEL> <text>"`.
    messages: Vec<String>,
    /// How many instances the layer reported itself active on.
    active: usize,
}

static KEPT: Mutex<Kept> = Mutex::new(Kept {
    messages: Vec::new(),
    active: 0,
});

fn kept() -> std::sync::MutexGuard<'static, Kept> {
    KEPT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The text and `log` target of a bridged `log` record.
#[derive(Default)]
struct Record {
    target: Option<String>,
    message: String,
}

impl Visit for Record {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "log.target" {
            self.target = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        }
    }
}

/// The subscriber's layer: keeps the messenger's records.
struct Collector;

impl<S: Subscriber> Layer<S> for Collector {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut record = Record::default();
        event.record(&mut record);
        if record.target.as_deref() != Some(MESSENGER_TARGET) {
            return;
        }
        let level = *event.metadata().level();
        let mut kept = kept();
        if level <= Level::WARN {
            kept.messages.push(format!("{level} {}", record.message));
        } else if record.message.contains(LAYER_ACTIVE) {
            kept.active += 1;
        }
    }
}

/// Installs the subscriber once per process; panics when another subscriber or logger
/// already holds the process, since the layer's messages would then go unseen.
fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // Debug: wgpu-hal subscribes the messenger to INFO (the layer's status message,
        // logged at debug) and WARNING only when `log` allows them at instance creation.
        tracing_subscriber::registry()
            .with(Collector.with_filter(LevelFilter::DEBUG))
            .try_init()
            .expect("the validation layer's collector needs the process's logger (R-16)");
    });
}

/// The guard of one `gpu` test (Q-T121.1): made before the test's first device, it installs
/// the collector, and when dropped in a debug build it panics on any warning or error the
/// Vulkan validation layer reported that nobody took, so the test fails (§1.13, R-16).
///
/// Messages are kept per process, not per thread: under `--test-threads 1`, the `gpu`
/// lane's rule (D-28), they are the guarded test's own.
#[must_use = "the guard checks the layer's messages when it is dropped"]
pub struct LayerMessages {
    _private: (),
}

/// Starts a `gpu` test's watch on the Vulkan validation layer (Q-T121.1): call it before
/// the test makes its first device, so the messenger subscribes to warnings.
pub fn layer_messages() -> LayerMessages {
    install();
    LayerMessages { _private: () }
}

impl LayerMessages {
    /// Takes the layer's warnings and errors so far, each as `"<LEVEL> <text>"`.
    pub fn take(&self) -> Vec<String> {
        std::mem::take(&mut kept().messages)
    }

    /// Takes how many instances the layer has reported itself active on (its status
    /// message at instance creation) since the last call.
    pub fn take_active(&self) -> usize {
        std::mem::take(&mut kept().active)
    }
}

impl Drop for LayerMessages {
    fn drop(&mut self) {
        // §1.13: a validation error in a debug run of the test suite is a test failure.
        if cfg!(debug_assertions) && !std::thread::panicking() {
            let left = self.take();
            if !left.is_empty() {
                panic!(
                    "{} Vulkan validation layer message(s): {left:#?}",
                    left.len()
                );
            }
        }
    }
}
