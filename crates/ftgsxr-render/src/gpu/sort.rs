// SPDX-License-Identifier: Apache-2.0

//! The radix sort of §1.4 stage 4 (E8, D-33, §2.4 GPU data layout): the player's own
//! least-significant-digit sort of (k̄, slot) pairs in `sort_count.wgsl`, `sort_scan.wgsl`
//! and `sort_scatter.wgsl`, run by `SplatPasses::encode_prepare` after stage 3.

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};

use super::frame::FrameSet;
use super::tier::SortKey;

/// `sort_count.wgsl`: one block's histogram of the pass's digit.
pub(crate) const SORT_COUNT_WGSL: &str = include_str!("shaders/sort_count.wgsl");
/// `sort_scan.wgsl`: one workgroup scans every block's histogram into scatter offsets.
pub(crate) const SORT_SCAN_WGSL: &str = include_str!("shaders/sort_scan.wgsl");
/// `sort_scatter.wgsl`: one block's stable scatter into the other buffer pair.
pub(crate) const SORT_SCATTER_WGSL: &str = include_str!("shaders/sort_scatter.wgsl");

/// The keys one workgroup of the count and scatter passes takes, four per thread of its
/// 256: stage 3 writes ⌈count / SORT_BLOCK⌉ workgroups (R-4), and the blocks' histograms
/// take 1 KiB per block, about C bytes (§1.12, Q-23). The kernels' override of the same
/// name. // Q-85
pub(crate) const SORT_BLOCK: u32 = 1024;
/// 8-bit digits (E8, D-33).
const RADIX: u64 = 256;
/// One pass's uniform block (the kernels' `Params`).
const PARAMS_BYTES: u64 = 16;
/// The passes' blocks sit 256 bytes apart, the largest uniform offset alignment wgpu allows.
const PARAMS_STRIDE: u64 = 256;

/// The bytes of the blocks' histograms for `capacity` keys: 256 32-bit bins per block.
pub(crate) fn histogram_bytes(capacity: u32) -> u64 {
    u64::from(capacity.div_ceil(SORT_BLOCK).max(1)) * RADIX * 4
}

/// The digits' shifts, least significant first: four 8-bit passes over k̄, or the top two
/// for `Depth16` (E8, D-33). Both counts are even, so the sorted pairs end in pair 0,
/// where they started.
fn shifts(key: SortKey) -> &'static [u32] {
    match key {
        SortKey::Depth32 => &[0, 8, 16, 24],
        SortKey::Depth16 => &[16, 24],
    }
}

/// Stage 4's pipelines, compiled once with the passes (§1.4, D-33).
pub(crate) struct RadixSort {
    device: wgpu::Device,
    layout: wgpu::BindGroupLayout,
    count: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    shifts: &'static [u32],
}

fn entry(binding: u32, ty: wgpu::BufferBindingType, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RadixSort {
    /// The three kernels for `key`'s passes on `device`; errors surface in the caller's
    /// validation scope.
    pub(crate) fn new(device: &wgpu::Device, key: SortKey) -> RadixSort {
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr sort"),
            entries: &[
                entry(0, read, false),
                entry(1, read, false),
                entry(2, write, false),
                entry(3, write, false),
                entry(4, write, false),
                entry(5, read, false),
                entry(6, wgpu::BufferBindingType::Uniform, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ftgsxr sort"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let constants = [("SORT_BLOCK", f64::from(SORT_BLOCK))];
        let pipeline = |name: &str, source: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let shifts = shifts(key);
        let params = device.create_buffer(&BufferDescriptor {
            label: Some("ftgsxr sort passes"),
            size: PARAMS_STRIDE * 4,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        let mut bytes = vec![0u8; (PARAMS_STRIDE * 4) as usize];
        for (i, shift) in shifts.iter().enumerate() {
            let at = i * PARAMS_STRIDE as usize;
            bytes[at..at + 4].copy_from_slice(&shift.to_le_bytes());
        }
        params
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&bytes);
        params.unmap();
        RadixSort {
            device: device.clone(),
            count: pipeline("ftgsxr sort_count.wgsl", SORT_COUNT_WGSL, "count"),
            scan: pipeline("ftgsxr sort_scan.wgsl", SORT_SCAN_WGSL, "scan"),
            scatter: pipeline("ftgsxr sort_scatter.wgsl", SORT_SCATTER_WGSL, "scatter"),
            layout,
            params,
            shifts,
        }
    }

    /// Stage 4 in one compute pass: for each digit, count, scan and scatter from one pair
    /// of keys and payloads into the other, every dispatch indirect from stage 3's
    /// arguments in `set.dispatch` (the blocks at word 0, the scan's one workgroup at word
    /// 3). The pairs start and end in `keys[0]` and `payloads[0]`.
    pub(crate) fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        set: &FrameSet,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let group = |from: usize| {
            let to = 1 - from;
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ftgsxr sort"),
                layout: &self.layout,
                entries: &[
                    whole(0, &set.keys[from]),
                    whole(1, &set.payloads[from]),
                    whole(2, &set.keys[to]),
                    whole(3, &set.payloads[to]),
                    whole(4, &set.histograms),
                    whole(5, &set.indirect),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: 0,
                            size: wgpu::BufferSize::new(PARAMS_BYTES),
                        }),
                    },
                ],
            })
        };
        let groups = [group(0), group(1)];
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ftgsxr sort"),
            timestamp_writes,
        });
        for i in 0..self.shifts.len() {
            let offset = (i as u64 * PARAMS_STRIDE) as u32;
            pass.set_bind_group(0, &groups[i % 2], &[offset]);
            pass.set_pipeline(&self.count);
            pass.dispatch_workgroups_indirect(&set.dispatch, 0);
            pass.set_pipeline(&self.scan);
            pass.dispatch_workgroups_indirect(&set.dispatch, SCAN_DISPATCH_OFFSET);
            pass.set_pipeline(&self.scatter);
            pass.dispatch_workgroups_indirect(&set.dispatch, 0);
        }
    }
}

/// The byte offset of the scan's dispatch arguments in `FrameSet::dispatch`.
const SCAN_DISPATCH_OFFSET: u64 = 12;

fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::SplatDevice;
    use crate::gpu::frame::FrameBuffers;
    use crate::gpu::readback::read_buffers;
    use crate::gpu::tier::QualityTier;

    /// SplitMix64, a seeded generator for the keys.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 32) as u32
        }
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// `keys` with payloads 0, 1, … sorted by `RadixSort` alone in a set of `capacity`
    /// slots whose slots past the keys hold key 0 (which must not be sorted), with stage
    /// 3's count and dispatch written by hand. Returns the first pair's keys and payloads.
    fn sort_on_gpu(
        dev: &SplatDevice,
        key: SortKey,
        keys: &[u32],
        capacity: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let n = keys.len() as u32;
        assert!(n <= capacity);
        let frame = FrameBuffers::new(dev, &QualityTier::conformance(), capacity);
        let set = &frame.set;
        let q = dev.queue();
        let mut padded = keys.to_vec();
        padded.resize(capacity as usize, 0);
        let mut payloads: Vec<u32> = (0..n).collect();
        payloads.resize(capacity as usize, u32::MAX);
        q.write_buffer(&set.keys[0], 0, &bytes(&padded));
        q.write_buffer(&set.payloads[0], 0, &bytes(&payloads));
        let mut counters = [0u32; 12];
        counters[1] = n;
        q.write_buffer(&set.indirect, 0, &bytes(&counters));
        let blocks = n.div_ceil(SORT_BLOCK);
        q.write_buffer(
            &set.dispatch,
            0,
            &bytes(&[blocks, 1, 1, blocks.min(1), 1, 1]),
        );
        let sort = RadixSort::new(dev.device(), key);
        let mut enc = dev
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        sort.encode(&mut enc, set, None);
        q.submit([enc.finish()]);
        let out = read_buffers(
            dev,
            &[
                (&set.keys[0], u64::from(n) * 4),
                (&set.payloads[0], u64::from(n) * 4),
            ],
        );
        assert!(dev.take_validation_messages().is_empty());
        (words(&out[0]), words(&out[1]))
    }

    /// The stable CPU sort the GPU must equal: the payloads 0, 1, … ordered by `digits`
    /// of their keys, ties in payload order.
    fn stable_order(keys: &[u32], digits: impl Fn(u32) -> u32) -> Vec<u32> {
        let mut order: Vec<u32> = (0..keys.len() as u32).collect();
        order.sort_by_key(|&i| digits(keys[i as usize]));
        order
    }

    fn check(dev: &SplatDevice, key: SortKey, keys: &[u32], capacity: u32) {
        let (got_keys, got_payloads) = sort_on_gpu(dev, key, keys, capacity);
        let expected = match key {
            SortKey::Depth32 => stable_order(keys, |k| k),
            SortKey::Depth16 => stable_order(keys, |k| k >> 16),
        };
        assert_eq!(got_payloads, expected, "{key:?}, {} keys", keys.len());
        let moved: Vec<u32> = expected.iter().map(|&i| keys[i as usize]).collect();
        assert_eq!(got_keys, moved, "the keys travel with their payloads");
    }

    /// Random keys, a partial last block, an exact block multiple, a single key and no key,
    /// with stale keys past the count left unsorted.
    #[test]
    #[ignore = "gpu"]
    fn gpu_radix_sort_orders_keys_stably_with_any_tail() {
        let dev = SplatDevice::headless().expect("a graphics adapter for the gpu lane");
        let mut rng = Rng(0x113);
        for (n, capacity) in [(3000, 5000), (2048, 2048), (1, 300), (0, 10), (1023, 4096)] {
            let keys: Vec<u32> = (0..n).map(|_| rng.next() | 1).collect();
            check(&dev, SortKey::Depth32, &keys, capacity);
        }
    }

    /// Equal keys keep their order, within a block and across blocks: all equal, and few
    /// distinct values over many blocks.
    #[test]
    #[ignore = "gpu"]
    fn gpu_radix_sort_is_stable_on_repeated_keys() {
        let dev = SplatDevice::headless().expect("a graphics adapter for the gpu lane");
        check(&dev, SortKey::Depth32, &[0xdead_beef; 5000], 5000);
        let mut rng = Rng(7);
        let values: Vec<u32> = (0..5).map(|_| rng.next()).collect();
        let keys: Vec<u32> = (0..50_000)
            .map(|_| values[rng.next() as usize % values.len()])
            .collect();
        check(&dev, SortKey::Depth32, &keys, 50_000);
    }

    /// `Depth16` sorts by the top 16 bits only, stably (E8).
    #[test]
    #[ignore = "gpu"]
    fn gpu_radix_sort_depth16_orders_by_the_top_16_bits() {
        let dev = SplatDevice::headless().expect("a graphics adapter for the gpu lane");
        let mut rng = Rng(16);
        let keys: Vec<u32> = (0..100_000).map(|_| rng.next() & 0x00ff_ffff).collect();
        check(&dev, SortKey::Depth16, &keys, 100_000);
    }

    /// 10⁶ random keys, the size of R-4.
    #[test]
    #[ignore = "gpu"]
    fn gpu_radix_sort_orders_a_million_keys() {
        let dev = SplatDevice::headless().expect("a graphics adapter for the gpu lane");
        let mut rng = Rng(1_000_000);
        let keys: Vec<u32> = (0..1_000_000).map(|_| rng.next()).collect();
        check(&dev, SortKey::Depth32, &keys, 1_000_000);
    }

    fn validate(name: &str, source: &str, entry: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} parses: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validates without capabilities: {e:?}"));
        assert!(
            module.entry_points.iter().any(|e| e.name == entry),
            "{name}: entry point {entry}"
        );
    }

    /// The three kernels are valid WGSL that needs no capability (no subgroup operations,
    /// D-33).
    #[test]
    fn radix_sort_shaders_are_valid_wgsl_without_capabilities() {
        validate("sort_count.wgsl", SORT_COUNT_WGSL, "count");
        validate("sort_scan.wgsl", SORT_SCAN_WGSL, "scan");
        validate("sort_scatter.wgsl", SORT_SCATTER_WGSL, "scatter");
    }
}
