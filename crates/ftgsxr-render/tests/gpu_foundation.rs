// SPDX-License-Identifier: Apache-2.0

//! The GPU foundation of `ftgsxr_render::gpu` (`gpu` lane, #111): `GpuModel`'s static bytes
//! against §1.12's table, the memory check before upload (§2.4 rule 4, D-13), the desktop
//! budget (Q-66) and the validation collector (§1.13, the plumbing R-16 relies on).

use ftgsxr_core::{GpuLayoutOptions, HalfFields, TemporalIndex, pack_for_gpu};
use ftgsxr_render::{BudgetSource, GpuModel, MemoryBudget, RenderError, SplatDevice};
use ftgsxr_testkit::{gpu_context, random_model};

const GIB: f64 = (1u64 << 30) as f64;

/// Packs and uploads a random model of `n` rows at degree `degree` under `opts`.
fn upload(
    dev: &SplatDevice,
    n: usize,
    degree: u8,
    opts: &GpuLayoutOptions,
) -> Result<GpuModel, RenderError> {
    let model = random_model(7 + u64::from(degree), n, degree);
    let (index, order) = TemporalIndex::build(&model);
    let arrays = pack_for_gpu(&model, &order, opts);
    GpuModel::upload(dev, &arrays, index, &model.conventions)
}

#[test]
#[ignore = "gpu"]
fn gpu_memory_bytes_equal_the_static_bytes_of_section_1_12() {
    // §1.12 Memory: float32 storage 76, 112, 172, 256 bytes per primitive for L = 0 … 3;
    // half-precision storage 48, 66, 96, 138.
    let float32 = [76u64, 112, 172, 256];
    let half = [48u64, 66, 96, 138];
    let half_opts = GpuLayoutOptions {
        half: HalfFields {
            velocity: true,
            scale: true,
            rotation: true,
            opacity: true,
            sh: true,
        },
        sh_cap: None,
    };
    let dev = gpu_context();
    let n = 64;
    for degree in 0..=3u8 {
        let l = usize::from(degree);
        let m = upload(&dev, n, degree, &GpuLayoutOptions::default()).expect("fits the budget");
        assert_eq!(m.n(), n as u32);
        assert_eq!(
            m.memory_bytes(),
            n as u64 * float32[l],
            "float32, L = {degree}"
        );
        let m = upload(&dev, n, degree, &half_opts).expect("fits the budget");
        assert_eq!(m.memory_bytes(), n as u64 * half[l], "half, L = {degree}");
    }
    assert!(dev.take_validation_messages().is_empty());
}

#[test]
#[ignore = "gpu"]
fn gpu_desktop_budget_on_the_rtx_4090_lies_between_15_and_18_5_gib() {
    // Q-66's detecting test: 75 % of the RTX 4090's 24 GiB device-local heap is 18 GiB.
    // The bounds are read in GiB (2³⁰ bytes): the card's 24 "GB" are 24 GiB, so 75 % of
    // them is 19.3 × 10⁹ bytes, outside a decimal 18.5 GB.
    let dev = gpu_context();
    let budget = dev.memory_budget();
    assert!(
        matches!(budget.source, BudgetSource::AdapterHeap { .. }),
        "the Vulkan heap query fell back: {budget:?}"
    );
    assert!(
        dev.adapter_name().contains("4090"),
        "the reference GPU of §1.1 is an RTX 4090, this adapter is {}",
        dev.adapter_name()
    );
    let gib = budget.bytes as f64 / GIB;
    assert!((15.0..=18.5).contains(&gib), "{gib} GiB");
}

#[test]
#[ignore = "gpu"]
fn gpu_model_over_the_budget_is_refused_before_upload() {
    let mut dev = gpu_context();
    // §1.12: 64 rows at L = 3 in float32 take 64 · 256 = 16 384 bytes.
    let needed = 64 * 256;
    dev.set_memory_budget(MemoryBudget {
        bytes: needed - 1,
        source: BudgetSource::Tier,
    });
    match upload(&dev, 64, 3, &GpuLayoutOptions::default()) {
        Err(RenderError::Budget { needed: n, budget }) => {
            assert_eq!(n, needed);
            assert_eq!(budget, needed - 1);
        }
        Err(e) => panic!("expected E-MEM-BUDGET, got {e:?}"),
        Ok(_) => panic!("a model over the budget was uploaded"),
    }
    // At the budget exactly it fits.
    dev.set_memory_budget(MemoryBudget {
        bytes: needed,
        source: BudgetSource::Tier,
    });
    assert!(upload(&dev, 64, 3, &GpuLayoutOptions::default()).is_ok());
}

/// A call wgpu's validation rejects: a buffer both mappable for reading and a storage
/// buffer, which WebGPU forbids without `MAPPABLE_PRIMARY_BUFFERS`.
fn invalid_call(dev: &SplatDevice) {
    let _ = dev.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("deliberately invalid"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
}

#[test]
#[ignore = "gpu"]
fn gpu_collector_catches_a_deliberately_invalid_call() {
    let dev = gpu_context();
    assert!(dev.take_validation_messages().is_empty());
    invalid_call(&dev);
    let messages = dev.take_validation_messages();
    assert_eq!(messages.len(), 1, "{messages:?}");
    assert!(dev.take_validation_messages().is_empty(), "taking clears");
}

// The check is a debug build's (§1.13).
#[cfg(debug_assertions)]
#[test]
#[ignore = "gpu"]
#[should_panic(expected = "GPU validation message")]
fn gpu_collector_fails_the_test_that_made_an_invalid_call() {
    // §1.13: a validation error in a debug run of the test suite is a test failure; the
    // device panics when it is dropped with messages nobody took.
    let dev = gpu_context();
    invalid_call(&dev);
    drop(dev);
}
