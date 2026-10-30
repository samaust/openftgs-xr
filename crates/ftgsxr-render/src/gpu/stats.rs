// SPDX-License-Identifier: Apache-2.0

//! Stage 8 of §1.4: `FrameStats`, read two frames late without blocking (D-34, P-5).

use std::sync::atomic::Ordering;

use super::device::SplatDevice;
use super::frame::{FrameBuffers, SLOT_FREE, SLOT_READY, STAGING_SLOTS, STATS_BYTES};

/// One frame's statistics for the HUD (§1.13): the counts after stage 3's clamp, C, the
/// overflow flag, and the GPU times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    /// Splat records drawn, at most C.
    pub visible_splats: u32,
    /// Center records drawn, at most C less the splats (Q-T112.1).
    pub visible_centers: u32,
    /// C.
    pub capacity: u32,
    /// Whether rows were dropped for want of capacity (D-34).
    pub overflow: bool,
    /// The GPU times of the frame's passes.
    pub gpu_times_ms: GpuTimes,
}

/// GPU times in milliseconds from timestamp queries (§1.13); 0 where the device has no
/// timestamp queries or the pass wrote none that frame (Q-T112.4).
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
    /// The statistics `encode_stats` copied two frames ago (frame n − 2 after the n-th
    /// call, counting from 0), once their mapping has completed; `None` before that or
    /// when that frame's copy was skipped. Polls the device without waiting: it never
    /// blocks (P-5, Q-T112.4).
    pub fn poll(frame: &FrameBuffers, dev: &SplatDevice) -> Option<FrameStats> {
        // Q-T112.4
        let _ = dev.device().poll(wgpu::PollType::Poll);
        let ring = &frame.ring;
        let encoded = ring.encoded.load(Ordering::Acquire);
        let target = encoded.checked_sub(STAGING_SLOTS as u64)?;
        let slot = (target % STAGING_SLOTS as u64) as usize;
        if ring.frame[slot].load(Ordering::Acquire) != target
            || ring.state[slot].load(Ordering::Acquire) != SLOT_READY
        {
            return None;
        }
        let staging = &frame.set.staging[slot];
        let mut bytes = [0u8; STATS_BYTES as usize];
        bytes.copy_from_slice(&staging.get_mapped_range(..));
        staging.unmap();
        ring.state[slot].store(SLOT_FREE, Ordering::Release);
        ring.frame[slot].store(u64::MAX, Ordering::Release);
        Some(decode_stats(&bytes, dev.queue().get_timestamp_period()))
    }
}

/// The statistics block (`cull.wgsl`'s `size`, Q-T112.4): the two counts, the overflow
/// flag and C in words 0–3, then six 64-bit timestamps (the cull's begin and end, the
/// sort's, the draw's) in ticks of `period_ns`.
pub(crate) fn decode_stats(bytes: &[u8; STATS_BYTES as usize], period_ns: f32) -> FrameStats {
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("4 bytes"));
    let ts: [u64; 6] = std::array::from_fn(|i| {
        u64::from_le_bytes(bytes[16 + 8 * i..24 + 8 * i].try_into().expect("8 bytes"))
    });
    let ms = |ticks: u64| (ticks as f64 * f64::from(period_ns) / 1e6) as f32;
    let span = |a: u64, b: u64| {
        if a != 0 && b >= a { ms(b - a) } else { 0.0 }
    };
    let last = ts.iter().copied().filter(|&t| t != 0).max().unwrap_or(0);
    FrameStats {
        visible_splats: word(0),
        visible_centers: word(1),
        overflow: word(2) != 0,
        capacity: word(3),
        gpu_times_ms: GpuTimes {
            cull: span(ts[0], ts[1]),
            sort: span(ts[2], ts[3]),
            draw: span(ts[4], ts[5]),
            total: span(ts[0], last),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(words: [u32; 4], ts: [u64; 6]) -> [u8; STATS_BYTES as usize] {
        let mut b = [0u8; STATS_BYTES as usize];
        for (i, w) in words.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        for (i, t) in ts.iter().enumerate() {
            b[16 + 8 * i..24 + 8 * i].copy_from_slice(&t.to_le_bytes());
        }
        b
    }

    #[test]
    fn decode_stats_reads_the_counts_and_the_flag() {
        let s = decode_stats(&block([7, 3, 1, 10], [0; 6]), 1.0);
        assert_eq!(
            (s.visible_splats, s.visible_centers, s.overflow, s.capacity),
            (7, 3, true, 10)
        );
        assert_eq!(s.gpu_times_ms, GpuTimes::default());
    }

    #[test]
    fn decode_stats_turns_ticks_into_milliseconds() {
        // 1 ns per tick: the cull 2 ms, no sort or draw written, total 2 ms.
        let s = decode_stats(
            &block([0, 0, 0, 1], [1_000_000, 3_000_000, 0, 0, 0, 0]),
            1.0,
        );
        assert_eq!(s.gpu_times_ms.cull, 2.0);
        assert_eq!(s.gpu_times_ms.sort, 0.0);
        assert_eq!(s.gpu_times_ms.draw, 0.0);
        assert_eq!(s.gpu_times_ms.total, 2.0);
        // 2 ns per tick, every pass written.
        let s = decode_stats(
            &block([0; 4], [1_000, 2_000, 2_500, 3_000, 4_000, 6_000]),
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
}
