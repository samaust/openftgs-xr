// SPDX-License-Identifier: Apache-2.0

//! Stage 8 of §1.4: `FrameStats`, read two frames late without blocking (D-34, P-5).

use std::sync::atomic::Ordering;

use super::device::SplatDevice;
use super::frame::{
    FrameBuffers, SLOT_FREE, STAGING_SLOTS, STATS_BYTES, STATS_TIMESTAMPS, pick_block,
};

/// One frame's statistics for the HUD (§1.13): the counts after stage 3's clamp, C, the
/// overflow flag, the non-finite count, and the GPU times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    /// Splat records drawn, at most C.
    pub visible_splats: u32,
    /// Center records drawn, at most C less the splats (Q-78).
    pub visible_centers: u32,
    /// C.
    pub capacity: u32,
    /// Whether rows were dropped for want of capacity (D-34).
    pub overflow: bool,
    /// Primitives culled for a non-finite value in their projection (E6), over every
    /// instance drawn by the view (Q-T188.1).
    pub non_finite: u32,
    /// The GPU times of the frame's passes.
    pub gpu_times_ms: GpuTimes,
}

/// GPU times in milliseconds from timestamp queries (§1.13); 0 where the device has no
/// timestamp queries or the pass wrote none that frame (Q-81).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTimes {
    /// Stages 1–3: the cull and size dispatches.
    pub cull: f32,
    /// Stage 4.
    pub sort: f32,
    /// Stages 5 and 6.
    pub draw: f32,
    /// From the first timestamp to the last one written.
    pub total: f32,
}

impl FrameStats {
    /// The newest block `encode_stats` copied two or more frames ago (frame n − 2 or
    /// earlier after the n-th call, counting from 0) whose mapping has completed; the older
    /// mapped blocks it passes over are dropped. `None` when no such block is mapped. Each
    /// block is returned once, and on a GPU that keeps pace it is frame n − 2's. Polls the
    /// device without waiting: it never blocks (P-5, Q-81).
    pub fn poll(frame: &FrameBuffers, dev: &SplatDevice) -> Option<FrameStats> {
        // Q-81
        let _ = dev.device().poll(wgpu::PollType::Poll);
        let ring = &frame.ring;
        let pick = pick_block(
            ring.encoded.load(Ordering::Acquire),
            &ring.states(),
            &ring.frames(),
        )?;
        let mut bytes = [0u8; STATS_BYTES as usize];
        bytes.copy_from_slice(&frame.set.staging[pick.take].get_mapped_range(..));
        for slot in 0..STAGING_SLOTS {
            if pick.released[slot] {
                frame.set.staging[slot].unmap();
                ring.state[slot].store(SLOT_FREE, Ordering::Release);
                ring.frame[slot].store(u64::MAX, Ordering::Release);
            }
        }
        Some(decode_stats(&bytes, dev.queue().get_timestamp_period()))
    }
}

/// The statistics block (`cull.wgsl`'s `size`, §1.4 stage 8, Q-81): the two counts, the
/// overflow flag, C and the non-finite count in words 0–4, three pad words, then from
/// byte `STATS_TIMESTAMPS` six 64-bit timestamps (the cull's begin and end, the sort's,
/// the draw's) in ticks of `period_ns`, 0 where the frame wrote none.
pub(crate) fn decode_stats(bytes: &[u8; STATS_BYTES as usize], period_ns: f32) -> FrameStats {
    // #193: each range is a fixed sub-slice of the STATS_BYTES block, in bounds and of
    // the array's length, so the conversions cannot fail.
    let word = |i: usize| {
        u32::from_le_bytes(
            bytes[4 * i..4 * i + 4]
                .try_into()
                .expect("invariant: a 4-byte range of the STATS_BYTES block converts to [u8; 4]"),
        )
    };
    let ts: [u64; 6] = std::array::from_fn(|i| {
        let at = STATS_TIMESTAMPS as usize + 8 * i;
        u64::from_le_bytes(
            bytes[at..at + 8]
                .try_into()
                .expect("invariant: an 8-byte range of the STATS_BYTES block converts to [u8; 8]"),
        )
    });
    let ms = |ticks: u64| (ticks as f64 * f64::from(period_ns) / 1e6) as f32;
    let span = |a: u64, b: u64| {
        if a != 0 && b >= a { ms(b - a) } else { 0.0 }
    };
    // Q-115: from the first timestamp written to the last, whichever passes ran.
    let written = || ts.iter().copied().filter(|&t| t != 0);
    let first = written().min().unwrap_or(0);
    let last = written().max().unwrap_or(0);
    FrameStats {
        visible_splats: word(0),
        visible_centers: word(1),
        overflow: word(2) != 0,
        capacity: word(3),
        non_finite: word(4),
        gpu_times_ms: GpuTimes {
            cull: span(ts[0], ts[1]),
            sort: span(ts[2], ts[3]),
            draw: span(ts[4], ts[5]),
            total: span(first, last),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(words: [u32; 5], ts: [u64; 6]) -> [u8; STATS_BYTES as usize] {
        let mut b = [0u8; STATS_BYTES as usize];
        for (i, w) in words.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        for (i, t) in ts.iter().enumerate() {
            let at = STATS_TIMESTAMPS as usize + 8 * i;
            b[at..at + 8].copy_from_slice(&t.to_le_bytes());
        }
        b
    }

    #[test]
    fn decode_stats_reads_the_counts_and_the_flag() {
        let s = decode_stats(&block([7, 3, 1, 10, 0], [0; 6]), 1.0);
        assert_eq!(
            (s.visible_splats, s.visible_centers, s.overflow, s.capacity),
            (7, 3, true, 10)
        );
        assert_eq!(s.non_finite, 0);
        assert_eq!(s.gpu_times_ms, GpuTimes::default());
    }

    #[test]
    fn decode_stats_reads_the_non_finite_count_beside_the_timestamps() {
        // §1.4 stage 8, revision 13: the non-finite count in word 4; the timestamps after
        // the pad words are read unchanged by it.
        let s = decode_stats(
            &block([4, 0, 0, 10, 17], [1_000_000, 2_000_000, 0, 0, 0, 0]),
            1.0,
        );
        assert_eq!(s.non_finite, 17);
        assert_eq!((s.visible_splats, s.capacity, s.overflow), (4, 10, false));
        assert_eq!(s.gpu_times_ms.cull, 1.0);
        assert_eq!(s.gpu_times_ms.total, 1.0);
    }

    #[test]
    fn decode_stats_turns_ticks_into_milliseconds() {
        // 1 ns per tick: the cull 2 ms, no sort or draw written, total 2 ms.
        let s = decode_stats(
            &block([0, 0, 0, 1, 0], [1_000_000, 3_000_000, 0, 0, 0, 0]),
            1.0,
        );
        assert_eq!(s.gpu_times_ms.cull, 2.0);
        assert_eq!(s.gpu_times_ms.sort, 0.0);
        assert_eq!(s.gpu_times_ms.draw, 0.0);
        assert_eq!(s.gpu_times_ms.total, 2.0);
        // 2 ns per tick, every pass written.
        let s = decode_stats(
            &block([0; 5], [1_000, 2_000, 2_500, 3_000, 4_000, 6_000]),
            2.0,
        );
        assert_eq!(
            [
                s.gpu_times_ms.cull,
                s.gpu_times_ms.sort,
                s.gpu_times_ms.draw,
                s.gpu_times_ms.total
            ],
            [0.002, 0.001, 0.004, 0.01]
        );
    }

    #[test]
    fn decode_stats_with_only_the_draw_written_times_the_draw_alone() {
        // A frame that skipped stages 1–4 (Q-100): the cull and sort wrote nothing, so
        // they are 0 (Q-81), and the total runs from the draw's begin to its end (Q-115).
        let s = decode_stats(
            &block([5, 0, 0, 10, 0], [0, 0, 0, 0, 7_000_000, 10_000_000]),
            1.0,
        );
        assert_eq!(s.gpu_times_ms.cull, 0.0);
        assert_eq!(s.gpu_times_ms.sort, 0.0);
        assert_eq!(s.gpu_times_ms.draw, 3.0);
        assert_eq!(s.gpu_times_ms.total, 3.0);
    }
}
