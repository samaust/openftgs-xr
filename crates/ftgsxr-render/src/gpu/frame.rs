// SPDX-License-Identifier: Apache-2.0

//! `FrameBuffers`: one set of a view's per-frame buffers, sized once per tier and
//! capacity (§1.4, §1.12 Memory, §2.4 rule 4 and GPU data layout, D-34).

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};

use super::device::SplatDevice;
use super::sort::histogram_bytes;
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
/// The counters (`cull.wgsl`): two indirect draws (splats and centers, §1.4 stages 5 and
/// 6) of four words each, the overflow flag, the rows that passed E3's test (Q-79) and
/// two spare words.
pub(crate) const COUNTER_WORDS: u64 = 12;
const INDIRECT_BYTES: u64 = COUNTER_WORDS * WORD;
/// The sort passes' indirect dispatches (§1.4 stage 3): the blocks of the count and
/// scatter passes, then the scan's one workgroup, three words each. // Q-85
const DISPATCH_BYTES: u64 = 6 * WORD;
/// The statistics block (§1.4 stage 8): the two counts, the overflow flag, C, and six
/// 64-bit timestamps (Q-81).
pub(crate) const STATS_BYTES: u64 = 64;
/// The statistics ring's staging buffers (§1.4 stage 8, P-5): one per frame, since
/// mapping is a whole buffer's state, and one more than the delay needs, so that a block
/// that maps late is kept (Q-81).
pub(crate) const STAGING_SLOTS: usize = 4;
/// Statistics are read two frames late (§1.4 stage 8): after the n-th `encode_stats`
/// (from 0), `poll` returns no block of frame n − 1 or n (Q-81).
pub(crate) const STATS_DELAY: u64 = 2;
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

impl StatsRing {
    /// Each buffer's state, as `claim_slot` and `pick_block` read it.
    pub(crate) fn states(&self) -> [u8; STAGING_SLOTS] {
        std::array::from_fn(|s| self.state[s].load(Ordering::Acquire))
    }

    /// The frame each buffer holds, `u64::MAX` for none.
    pub(crate) fn frames(&self) -> [u64; STAGING_SLOTS] {
        std::array::from_fn(|s| self.frame[s].load(Ordering::Acquire))
    }
}

/// What `encode_stats` does with its staging buffer (Q-81).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Claim {
    /// Free: copy the frame's block into it.
    Copy,
    /// Mapped but never returned: unmap it, then copy.
    Reuse,
    /// Still waiting for its mapping: skip the frame's copy, keeping the frame it holds.
    Skip,
}

/// The buffer the n-th `encode_stats` (from 0) takes, n mod `STAGING_SLOTS`, and what it
/// does with it (Q-81).
pub(crate) fn claim_slot(n: u64, state: &[u8; STAGING_SLOTS]) -> (usize, Claim) {
    let slot = (n % STAGING_SLOTS as u64) as usize;
    let claim = match state[slot] {
        SLOT_READY => Claim::Reuse,
        SLOT_PENDING => Claim::Skip,
        _ => Claim::Copy,
    };
    (slot, claim)
}

/// The block `poll` returns (Q-81): the buffer to read, and the older mapped buffers it
/// passes over, to unmap and free with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pick {
    pub(crate) take: usize,
    pub(crate) free: [bool; STAGING_SLOTS],
}

/// The newest mapped block of frame n − 2 or earlier, `encoded` being n + 1, and the older
/// mapped ones; `None` when no such block is mapped (Q-81).
pub(crate) fn pick_block(
    encoded: u64,
    state: &[u8; STAGING_SLOTS],
    frame: &[u64; STAGING_SLOTS],
) -> Option<Pick> {
    let newest = encoded.checked_sub(1 + STATS_DELAY)?;
    let mapped: [bool; STAGING_SLOTS] =
        std::array::from_fn(|s| state[s] == SLOT_READY && frame[s] <= newest);
    let take = (0..STAGING_SLOTS)
        .filter(|&s| mapped[s])
        .max_by_key(|&s| frame[s])?;
    let free = std::array::from_fn(|s| mapped[s] && s != take);
    Some(Pick { take, free })
}

/// The buffers of one set (§2.1's list); the passes bind them.
pub(crate) struct FrameSet {
    pub(crate) records: wgpu::Buffer,
    pub(crate) keys: [wgpu::Buffer; 2],
    pub(crate) payloads: [wgpu::Buffer; 2],
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
    /// E7's constants per instance of the last `encode_prepare` (α_max, α_skip, the pixel
    /// center offset; D-39), which the splat draw reads by each record's instance.
    pub(crate) draw_constants: Mutex<Option<wgpu::Buffer>>,
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
            // Q-23's "about C": 1 KiB per block of 1024 keys (Q-85).
            histograms: buffer(dev, "ftgsxr histograms", histogram_bytes(capacity), storage),
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
            staging: std::array::from_fn(staging),
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
            draw_constants: Mutex::new(None),
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
        ]
        .into_iter()
        .chain(&s.staging)
        .map(|b| b.size())
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_core::memory_estimate;

    /// The ring's state without a GPU: `encode` and `poll` apply `claim_slot` and
    /// `pick_block` as `encode_stats` and `FrameStats::poll` do, and `map` stands in for a
    /// buffer's mapping completing.
    struct Ring {
        state: [u8; STAGING_SLOTS],
        frame: [u64; STAGING_SLOTS],
        encoded: u64,
    }

    impl Ring {
        fn new() -> Ring {
            Ring {
                state: [SLOT_FREE; STAGING_SLOTS],
                frame: [u64::MAX; STAGING_SLOTS],
                encoded: 0,
            }
        }

        /// Encodes the next frame; the claim made.
        fn encode(&mut self) -> Claim {
            let n = self.encoded;
            self.encoded += 1;
            let (slot, claim) = claim_slot(n, &self.state);
            if claim != Claim::Skip {
                self.state[slot] = SLOT_PENDING;
                self.frame[slot] = n;
            }
            claim
        }

        /// Frame `f`'s buffer, if it is waiting for its mapping, is mapped.
        fn map(&mut self, f: u64) {
            for slot in 0..STAGING_SLOTS {
                if self.frame[slot] == f && self.state[slot] == SLOT_PENDING {
                    self.state[slot] = SLOT_READY;
                }
            }
        }

        /// The frame whose block `poll` returns.
        fn poll(&mut self) -> Option<u64> {
            let pick = pick_block(self.encoded, &self.state, &self.frame)?;
            let f = self.frame[pick.take];
            for slot in 0..STAGING_SLOTS {
                if slot == pick.take || pick.free[slot] {
                    assert_eq!(
                        self.state[slot], SLOT_READY,
                        "only mapped buffers are freed"
                    );
                    self.state[slot] = SLOT_FREE;
                    self.frame[slot] = u64::MAX;
                }
            }
            Some(f)
        }
    }

    #[test]
    fn pick_block_on_a_ring_that_keeps_pace_returns_frame_n_minus_2() {
        let mut ring = Ring::new();
        for n in 0..12u64 {
            assert_eq!(ring.encode(), Claim::Copy, "frame {n}");
            ring.map(n);
            let expected = n.checked_sub(2);
            assert_eq!(ring.poll(), expected, "after the {n}-th encode");
            assert_eq!(ring.poll(), None, "polled again after the {n}-th encode");
        }
    }

    #[test]
    fn pick_block_returns_a_block_of_frame_n_minus_3_that_maps_late() {
        let mut ring = Ring::new();
        for _ in 0..3 {
            ring.encode();
        }
        assert_eq!(ring.poll(), None, "frame 0 is not mapped yet");
        ring.encode();
        ring.map(0);
        assert_eq!(ring.poll(), Some(0), "frame 1's buffer is not mapped");
        assert_eq!(ring.poll(), None, "frame 0's block is returned once");
    }

    #[test]
    fn pick_block_returns_frame_n_minus_2_and_frees_frame_n_minus_3() {
        let mut ring = Ring::new();
        for _ in 0..4 {
            ring.encode();
        }
        ring.map(0);
        ring.map(1);
        let pick = pick_block(ring.encoded, &ring.state, &ring.frame).expect("a block");
        assert_eq!(ring.frame[pick.take], 1);
        let freed: Vec<u64> = (0..STAGING_SLOTS)
            .filter(|&s| pick.free[s])
            .map(|s| ring.frame[s])
            .collect();
        assert_eq!(freed, [0], "frame 0's buffer is freed");
        assert_eq!(ring.poll(), Some(1));
        assert_eq!(ring.poll(), None, "frame 0's block is never returned");
        assert_eq!(
            ring.encode(),
            Claim::Copy,
            "frame 4 takes frame 0's freed buffer"
        );
    }

    #[test]
    fn claim_slot_skips_a_copy_over_a_buffer_waiting_for_its_mapping_and_keeps_its_frame() {
        let mut ring = Ring::new();
        for _ in 0..4 {
            ring.encode();
        }
        assert_eq!(claim_slot(4, &ring.state), (0, Claim::Skip));
        assert_eq!(
            ring.encode(),
            Claim::Skip,
            "frame 4 over frame 0's pending buffer"
        );
        assert_eq!(ring.frame[0], 0, "the buffer keeps frame 0");
        ring.map(0);
        assert_eq!(ring.poll(), Some(0), "frame 0's block, mapped at last");
    }

    #[test]
    fn claim_slot_reuses_a_mapped_buffer_nobody_returned() {
        let mut ring = Ring::new();
        for n in 0..4 {
            ring.encode();
            ring.map(n);
        }
        assert_eq!(claim_slot(4, &ring.state), (0, Claim::Reuse));
    }

    #[test]
    fn pick_block_returns_no_block_twice_nor_one_of_frame_n_minus_1_or_n() {
        // Mappings complete in submission order, a pseudo-random number of frames at a time
        // (none to all), and the ring is polled after every encode.
        let mut ring = Ring::new();
        let mut mapped = 0u64;
        let mut seed = 0x177u32;
        let mut last = None;
        for _ in 0..2000 {
            ring.encode();
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let k = u64::from(seed >> 29);
            while mapped < ring.encoded && mapped < ring.encoded.saturating_sub(4) + k {
                ring.map(mapped);
                mapped += 1;
            }
            if let Some(f) = ring.poll() {
                let n = ring.encoded - 1;
                assert!(
                    f + STATS_DELAY <= n,
                    "frame {f} returned after the {n}-th encode"
                );
                assert!(last < Some(f), "frame {f} returned after frame {last:?}");
                last = Some(f);
            }
        }
        assert!(last.is_some());
    }

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
