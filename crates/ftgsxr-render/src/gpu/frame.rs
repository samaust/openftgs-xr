// SPDX-License-Identifier: Apache-2.0

//! `FrameBuffers`: one set of a view's per-frame buffers, sized once per tier and
//! capacity (§1.4, §1.12 Memory, §2.4 rule 4 and GPU data layout, D-34).

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
/// Two indirect draws (splats and centers), four words each (§1.4 stages 5 and 6).
const INDIRECT_BYTES: u64 = 2 * 4 * WORD;
/// The sort passes' indirect dispatch (§1.4 stage 3), three words.
const DISPATCH_BYTES: u64 = 3 * WORD;
/// The statistics block (§1.4 stage 8).
const STATS_BYTES: u64 = 64;
/// Statistics are read two frames late (§1.4 stage 8, P-5): three staging slots.
const STAGING_SLOTS: u64 = 3;

/// The buffers of one set (§2.1's list); the passes of later tickets bind them.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "bound by the passes of later M-2 tickets (§1.4 stages 2–8)"
    )
)]
pub(crate) struct FrameSet {
    pub(crate) records: wgpu::Buffer,
    pub(crate) keys: [wgpu::Buffer; 2],
    pub(crate) payloads: [wgpu::Buffer; 2],
    pub(crate) histograms: wgpu::Buffer,
    pub(crate) indirect: wgpu::Buffer,
    pub(crate) dispatch: wgpu::Buffer,
    pub(crate) stats: wgpu::Buffer,
    pub(crate) staging: wgpu::Buffer,
}

/// One set of a view's frame buffers (§2.1): records, keys×2, payloads×2, histograms,
/// indirect arguments, dispatch arguments, statistics and staging, shared by every model
/// of the view (D-5); per-view mode holds one set per view.
pub struct FrameBuffers {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "bound by the passes of later M-2 tickets (§1.4 stages 2–8)"
        )
    )]
    set: FrameSet,
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

/// The record size of `tier`'s mode (§1.4 stage 2, §1.12).
fn record_bytes(tier: &QualityTier) -> u64 {
    match (tier.sort_mode, tier.half_records) {
        (SortMode::PerView, _) => VIEW_RECORD_BYTES,
        (SortMode::SharedCenter, false) => SHARED_RECORD_BYTES,
        (SortMode::SharedCenter, true) => HALF_SHARED_RECORD_BYTES,
    }
}

impl FrameBuffers {
    /// One set for `capacity` records in `tier`'s record layout (§2.4 rule 4: sized once
    /// per tier and capacity). A capacity of 0 still allocates one record's worth.
    pub fn new(dev: &SplatDevice, tier: &QualityTier, capacity: u32) -> FrameBuffers {
        let c = u64::from(capacity.max(1));
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let set = FrameSet {
            records: buffer(dev, "ftgsxr records", c * record_bytes(tier), storage),
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
            staging: buffer(
                dev,
                "ftgsxr statistics staging",
                STAGING_SLOTS * STATS_BYTES,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
        };
        FrameBuffers { set, capacity }
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
            &s.staging,
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
