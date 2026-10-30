// SPDX-License-Identifier: Apache-2.0

//! `SplatDevice`: the wgpu device the passes run on, its validation collector (§1.13) and
//! the memory budget uploads are checked against (§1.12, D-13, Q-66).

use std::sync::{Arc, Mutex, PoisonError};

use bevy::log::{info, warn};

use super::error::RenderError;
use super::tier::DESKTOP_FALLBACK_BUDGET;

/// Where a device's memory budget came from (Q-66), for the HUD and the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// 75 % of the adapter's largest device-local heap (§1.12's desktop rule).
    AdapterHeap {
        /// The heap's size in bytes.
        heap_bytes: u64,
    },
    /// The 8 GiB fallback, where the heap could not be queried.
    Fallback,
    /// A tier's fixed budget (the Quest's 1.5 GiB), set by the caller.
    Tier,
}

/// The static model bytes an upload may take, and where the figure came from (Q-66).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// The budget in bytes.
    pub bytes: u64,
    /// Where it came from.
    pub source: BudgetSource,
}

impl MemoryBudget {
    /// §1.12's desktop budget from the largest device-local heap, or the fallback (Q-66).
    pub(crate) fn desktop(heap_bytes: Option<u64>) -> MemoryBudget {
        match heap_bytes {
            // Q-66: 75 % of the heap.
            Some(heap_bytes) if heap_bytes > 0 => MemoryBudget {
                bytes: heap_bytes / 4 * 3,
                source: BudgetSource::AdapterHeap { heap_bytes },
            },
            _ => MemoryBudget {
                bytes: DESKTOP_FALLBACK_BUDGET,
                source: BudgetSource::Fallback,
            },
        }
    }
}

/// The GPU the passes run on (§2.1 `SplatDevice`): the device, its queue, limits and
/// features, the memory budget `GpuModel::upload` checks (D-13, Q-66), and the collector
/// of validation messages (§1.13, R-16).
///
/// The collector is the device's uncaptured-error handler: every validation, out-of-memory
/// or internal error wgpu reports is logged and kept until `take_validation_messages`.
/// In debug builds a `SplatDevice` dropped with messages nobody took panics, so a `gpu`
/// test fails on any (§1.13).
pub struct SplatDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    #[expect(
        dead_code,
        reason = "read by the passes of the later M-2 tickets (§2.1)"
    )]
    limits: wgpu::Limits,
    #[expect(
        dead_code,
        reason = "read by the passes of the later M-2 tickets (§2.1)"
    )]
    features: wgpu::Features,
    budget: MemoryBudget,
    messages: Arc<Mutex<Vec<String>>>,
}

/// Features the later passes use where the adapter offers them: timestamp queries for the
/// HUD's GPU times (§1.13) and indirect draws with a first instance (§1.4 stages 5 and 6).
fn wanted_features() -> wgpu::Features {
    wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::INDIRECT_FIRST_INSTANCE
}

impl SplatDevice {
    /// A device made elsewhere (the Bevy adapter's, §2.1). Installs the validation
    /// collector, replacing the device's uncaptured-error handler; the memory budget is
    /// Q-66's fallback until `set_memory_budget`.
    pub fn from_wgpu(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) -> SplatDevice {
        Self::build(device, queue, adapter_info, MemoryBudget::desktop(None))
    }

    /// A device of its own (Q-74): a Vulkan instance, created with the validation
    /// layers and the shader validator on in debug builds (§1.13), the high-performance
    /// adapter, and a device with the adapter's limits. The memory budget is §1.12's
    /// desktop rule (Q-66). `HeadlessRenderer` and `ftgsxr_testkit::gpu_context` use it.
    pub fn headless() -> Result<SplatDevice, RenderError> {
        // Q-74
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::VULKAN;
        // §1.13: validation in debug builds only (`debugging()` with debug assertions).
        desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
        let instance = wgpu::Instance::new(desc);
        let adapter =
            bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .map_err(|_| RenderError::NoAdapter)?;
        let required_features = adapter.features() & wanted_features();
        let (device, queue) =
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("ftgsxr splat device"),
                required_features,
                required_limits: adapter.limits(),
                ..Default::default()
            }))
            .map_err(|_| RenderError::NoAdapter)?;
        let budget = MemoryBudget::desktop(largest_device_local_heap(&adapter));
        match budget.source {
            BudgetSource::AdapterHeap { heap_bytes } => info!(
                "memory budget {} bytes: 75 % of the {heap_bytes}-byte device-local heap",
                budget.bytes
            ),
            _ => warn!(
                "memory budget {} bytes: the fallback, the adapter's heap could not be queried",
                budget.bytes
            ),
        }
        Ok(Self::build(device, queue, adapter.get_info(), budget))
    }

    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
        budget: MemoryBudget,
    ) -> SplatDevice {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&messages);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let text = e.to_string();
            warn!("GPU validation: {text}");
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(text);
        }));
        SplatDevice {
            limits: device.limits(),
            features: device.features(),
            device,
            queue,
            adapter_name: adapter_info.name,
            budget,
            messages,
        }
    }

    /// The adapter's name, as wgpu reports it.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The wgpu device (Q-68).
    pub fn device(&self) -> &wgpu::Device {
        // Q-68
        &self.device
    }

    /// The device's queue (Q-68).
    pub fn queue(&self) -> &wgpu::Queue {
        // Q-68
        &self.queue
    }

    /// The budget `GpuModel::upload` checks a model's static bytes against (D-13, Q-66).
    pub fn memory_budget(&self) -> MemoryBudget {
        self.budget
    }

    /// Replaces the budget, with a tier's fixed figure, say (Q-74).
    pub fn set_memory_budget(&mut self, budget: MemoryBudget) {
        self.budget = budget;
    }

    /// The validation messages collected since the last call, oldest first; taking them
    /// clears the collector (§1.13, R-16).
    pub fn take_validation_messages(&self) -> Vec<String> {
        std::mem::take(&mut *self.messages.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl Drop for SplatDevice {
    fn drop(&mut self) {
        // §1.13: a validation error in a debug run of the test suite is a failure. Q-74
        if cfg!(debug_assertions) && !std::thread::panicking() {
            let left = self.take_validation_messages();
            if !left.is_empty() {
                panic!("{} GPU validation message(s): {left:?}", left.len());
            }
        }
    }
}

/// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` (the Vulkan specification's `VkMemoryHeapFlagBits`).
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x1;

/// The size of the adapter's largest device-local memory heap, through wgpu-hal's Vulkan
/// adapter; `None` on another backend or with no such heap (Q-66).
fn largest_device_local_heap(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: `as_hal` requires that the hal adapter not be destroyed while wgpu uses it
    // and that wgpu-hal's rules hold: the guard is held only for this block, nothing is
    // destroyed or mutated, and `vkGetPhysicalDeviceMemoryProperties` only reads
    // properties of a physical device that the guard's instance enumerated and keeps
    // alive for the guard's lifetime.
    let props = unsafe {
        let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device())
    };
    props
        .memory_heaps_as_slice()
        .iter()
        .filter(|h| h.flags.as_raw() & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0)
        .map(|h| h.size)
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_desktop_is_three_quarters_of_the_heap() {
        let heap = 24 * (1u64 << 30);
        assert_eq!(
            MemoryBudget::desktop(Some(heap)),
            MemoryBudget {
                bytes: 18 * (1u64 << 30),
                source: BudgetSource::AdapterHeap { heap_bytes: heap }
            }
        );
    }

    #[test]
    fn memory_budget_desktop_falls_back_to_8_gib() {
        for heap in [None, Some(0)] {
            assert_eq!(
                MemoryBudget::desktop(heap),
                MemoryBudget {
                    bytes: 8 * (1u64 << 30),
                    source: BudgetSource::Fallback
                }
            );
        }
    }
}
