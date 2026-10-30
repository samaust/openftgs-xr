// SPDX-License-Identifier: Apache-2.0

//! `FrameBuffers`: one set of a view's per-frame buffers, sized once per tier and
//! capacity (§1.4, §1.12 Memory, §2.4 rule 4 and GPU data layout, D-34).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64};

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};

use super::device::SplatDevice;
use super::tier::{QualityTier, SortMode};

/// A per-view record (§1.4 stage 2): μ′, the conic, the half-extents, four half floats of
/// color and α, the depth and flags.
const VIEW_RECORD_BYTES: u64 = 48;
/// A shared-center record: the stage-space position, Σ^S, the color and the key.
const SHARED_RECORD_BYTES: u64 = 64;
/// A shared-center record with Σ^S in half precision (`half_records`, `quest-low`).
const HALF_SHARED_RECORD_BYTES: u64 = 48;
/// One key or payload, a 32-bit word (E8).
const WORD: u64 = 4;
/// The histograms' bytes per unit of capacity (Q-23: §1.12's "about C").
const HISTOGRAM_BYTES_PER_RECORD: u64 = 1;
/// The counters (`cull.wgsl`): two indirect draws (splats and centers, §1.4 stages 5 and
/// 6) of four words each, the overflow flag, the rows that passed E3's test (Q-79) and
/// two spare words.
pub(crate) const COUNTER_WORDS: u64 = 12;
const INDIRECT_BYTES: u64 = COUNTER_WORDS * WORD;
/// The sort passes' indirect dispatch (§1.4 stage 3), three words.
const DISPATCH_BYTES: u64 = 3 * WORD;
/// The statistics block (§1.4 stage 8): the two counts, the overflow flag, C, and six
/// 64-bit timestamps (Q-81).
pub(crate) const STATS_BYTES: u64 = 64;
/// Statistics are read two frames late (§1.4 stage 8, P-5): three staging buffers, since
/// mapping is a whole buffer's state.
pub(crate) const STAGING_SLOTS: usize = 3;
/// The timestamps a frame writes (Q-81): the cull pass's begin and end, the sort's,
/// the draw's.
pub(crate) const TIMESTAMPS: u32 = 6;

/// How the records of a set are laid out (`cull.wgsl`'s `RECORD_MODE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecordMode {
    /// 48-byte per-view records.
    PerView,
    /// 64-byte shared-center records.
    Shared,
    /// 48-byte shared-center records with Σ^S in half precision.
    SharedHalf,
}

impl RecordMode {
    pub(crate) fn of(tier: &QualityTier) -> RecordMode {
        match (tier.sort_mode, tier.half_records) {
            (SortMode::PerView, _) => RecordMode::PerView,
            (SortMode::SharedCenter, false) => RecordMode::Shared,
            (SortMode::SharedCenter, true) => RecordMode::SharedHalf,
        }
    }

    /// The record's bytes (§1.4 stage 2, §1.12).
    pub(crate) fn bytes(self) -> u64 {
        match self {
            RecordMode::PerView => VIEW_RECORD_BYTES,
            RecordMode::Shared => SHARED_RECORD_BYTES,
            RecordMode::SharedHalf => HALF_SHARED_RECORD_BYTES,
        }
    }

    /// `cull.wgsl`'s `RECORD_MODE`.
    pub(crate) fn shader_value(self) -> u32 {
        match self {
            RecordMode::PerView => 0,
            RecordMode::Shared => 1,
            RecordMode::SharedHalf => 2,
        }
    }
}

/// A staging buffer's state (§1.4 stage 8).
pub(crate) const SLOT_FREE: u8 = 0;
/// Copied into and waiting for its mapping, which follows the submission.
pub(crate) const SLOT_PENDING: u8 = 1;
/// Mapped and readable.
pub(crate) const SLOT_READY: u8 = 2;

/// The statistics ring (§1.4 stage 8, Q-81): which frame each staging buffer holds and
/// whether it is mapped, and how many frames `encode_stats` has encoded.
pub(crate) struct StatsRing {
    pub(crate) state: [Arc<AtomicU8>; STAGING_SLOTS],
    pub(crate) frame: [AtomicU64; STAGING_SLOTS],
    /// The frames encoded so far: the next frame's number.
    pub(crate) encoded: AtomicU64,
    /// The timestamps written since the last `encode_stats`, a prefix of the query set.
    pub(crate) written: AtomicU32,
}

/// The buffers of one set (§2.1's list); the passes bind them.
pub(crate) struct FrameSet {
    pub(crate) records: wgpu::Buffer,
    pub(crate) keys: [wgpu::Buffer; 2],
    pub(crate) payloads: [wgpu::Buffer; 2],
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "bound by the sort of #113 (§1.4 stage 4)")
    )]
    pub(crate) histograms: wgpu::Buffer,
    /// The counters, whose first eight words are the indirect draw arguments.
    pub(crate) indirect: wgpu::Buffer,
    pub(crate) dispatch: wgpu::Buffer,
    pub(crate) stats: wgpu::Buffer,
    pub(crate) staging: [wgpu::Buffer; STAGING_SLOTS],
    /// The frame's timestamps, where the device offers timestamp queries (§1.13).
    pub(crate) queries: Option<(wgpu::QuerySet, wgpu::Buffer)>,
}

/// One set of a view's frame buffers (§2.1): records, keys×2, payloads×2, histograms,
/// indirect arguments, dispatch arguments, statistics and staging, shared by every model
/// of the view (D-5); per-view mode holds one set per view.
pub struct FrameBuffers {
    pub(crate) set: FrameSet,
    pub(crate) mode: RecordMode,
    pub(crate) ring: StatsRing,
    capacity: u32,
}

fn buffer(dev: &SplatDevice, label: &str, size: u64, usage: BufferUsages) -> wgpu::Buffer {
    dev.device().create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.div_ceil(WORD).max(1) * WORD,
        usage,
        mapped_at_creation: false,
    })
}

impl FrameBuffers {
    /// One set for `capacity` records in `tier`'s record layout (§2.4 rule 4: sized once
    /// per tier and capacity). A capacity of 0 still allocates one record's worth.
    pub fn new(dev: &SplatDevice, tier: &QualityTier, capacity: u32) -> FrameBuffers {
        let c = u64::from(capacity.max(1));
        let mode = RecordMode::of(tier);
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let staging = |i: usize| {
            buffer(
                dev,
                &format!("ftgsxr statistics staging {i}"),
                STATS_BYTES,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
        };
        let queries = dev
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                let set = dev.device().create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("ftgsxr timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: TIMESTAMPS,
                });
                let resolve = buffer(
                    dev,
                    "ftgsxr timestamps resolve",
                    u64::from(TIMESTAMPS) * 8,
                    BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                );
                (set, resolve)
            });
        let set = FrameSet {
            records: buffer(dev, "ftgsxr records", c * mode.bytes(), storage),
            keys: [
                buffer(dev, "ftgsxr keys 0", c * WORD, storage),
                buffer(dev, "ftgsxr keys 1", c * WORD, storage),
            ],
            payloads: [
                buffer(dev, "ftgsxr payloads 0", c * WORD, storage),
                buffer(dev, "ftgsxr payloads 1", c * WORD, storage),
            ],
            histograms: buffer(
                dev,
                "ftgsxr histograms",
                c * HISTOGRAM_BYTES_PER_RECORD,
                storage,
            ),
            indirect: buffer(
                dev,
                "ftgsxr indirect draws",
                INDIRECT_BYTES,
                storage | BufferUsages::INDIRECT,
            ),
            dispatch: buffer(
                dev,
                "ftgsxr sort dispatch",
                DISPATCH_BYTES,
                storage | BufferUsages::INDIRECT,
            ),
            stats: buffer(dev, "ftgsxr statistics", STATS_BYTES, storage),
            staging: [staging(0), staging(1), staging(2)],
            queries,
        };
        let ring = StatsRing {
            state: std::array::from_fn(|_| Arc::new(AtomicU8::new(SLOT_FREE))),
            frame: std::array::from_fn(|_| AtomicU64::new(u64::MAX)),
            encoded: AtomicU64::new(0),
            written: AtomicU32::new(0),
        };
        FrameBuffers {
            set,
            mode,
            ring,
            capacity,
        }
    }

    /// C, the records the set holds.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The set's bytes on the GPU.
    #[cfg(test)]
    fn bytes(&self) -> u64 {
        let s = &self.set;
        [
            &s.records,
            &s.keys[0],
            &s.keys[1],
            &s.payloads[0],
            &s.payloads[1],
            &s.histograms,
            &s.indirect,
            &s.dispatch,
            &s.stats,
            &s.staging[0],
            &s.staging[1],
            &s.staging[2],
        ]
        .iter()
        .map(|b| b.size())
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_core::memory_estimate;

    /// Q-23: one set's bytes are the estimate's `per_capacity_bytes` times C, less 16 C for
    /// `quest-low`'s half-precision records, plus the arguments and statistics (under
    /// 1 KiB, §1.12).
    #[test]
    #[ignore = "gpu"]
    fn gpu_frame_buffers_bytes_match_the_estimate_for_every_tier() {
        // Not the testkit's `gpu_context`: a unit test links its own copy of this crate.
        let dev = SplatDevice::headless().expect("a graphics adapter for the gpu lane");
        let c: u32 = 1000;
        for tier in [
            QualityTier::conformance(),
            QualityTier::desktop(),
            QualityTier::quest_high(),
            QualityTier::quest_balanced(),
            QualityTier::quest_low(),
        ] {
            let shared = tier.sort_mode == SortMode::SharedCenter;
            let est = memory_estimate(1, 0, &tier.layout, shared);
            let per_c = u64::from(est.per_capacity_bytes) - if tier.half_records { 16 } else { 0 };
            let frame = FrameBuffers::new(&dev, &tier, c);
            assert_eq!(frame.capacity(), c);
            let extra = frame.bytes() - per_c * u64::from(c);
            assert!(
                extra < 1024,
                "{}: {extra} bytes over the estimate",
                tier.name
            );
        }
    }
}
