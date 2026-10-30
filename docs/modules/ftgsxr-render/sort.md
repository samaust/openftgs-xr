# The radix sort in `ftgsxr-render`

How `ftgsxr_render::gpu` sorts a view's records into draw order: stage 4 of §1.4's per-frame pipeline, the player's own radix sort of D-33, run inside `SplatPasses::encode_prepare`. The spec items are §1.4 stages 3 and 4, E8, D-33, D-34 and §2.4's GPU data layout, with R-4 as their test; this doc explains them and never overrides them.

## What is sorted

The cull pass (stage 2) gives each visible splat record a slot and writes, at that slot, the record's sort key k̄ (E8, always the full 32 bits, also in a `Depth16` tier) into the first key buffer and the slot itself into the first payload buffer. Stage 3 clamps the visible count to the capacity C and writes the sort's dispatch arguments. Stage 4 then sorts the (k̄, slot) pairs by ascending k̄, far to near, stably, so records of equal key keep their slot order (E8). The splat draw (stage 5) reads the sorted slots from the first payload buffer. Center records are never sorted: they follow the splats in the record buffer (Q-78) and stage 4 reads only the splat count.

## The passes

A least-significant-digit radix sort with 8-bit digits: four passes for `Depth32`, from the lowest digit up, and two for `Depth16`, over the top two digits only, which is E8's 16-bit key without a second key buffer. Each pass runs three kernels:

- **Count** (`sort_count.wgsl`). Each workgroup takes one block of consecutive keys, counts the pass's digit in a histogram in workgroup memory, and writes all 256 bins of its block (zeros included, so nothing needs clearing) to the histogram buffer.
- **Scan** (`sort_scan.wgsl`). One workgroup turns the blocks' histograms in place into each (block, digit)'s first output position: thread d walks digit d's column over the blocks, the 256 digit totals are scanned in workgroup memory, and each column gets its digit's base (§2.4's single-workgroup scan).
- **Scatter** (`sort_scatter.wgsl`). Each workgroup takes the same block again and moves every key and its payload to its output position in the other buffer pair.

Keys and payloads ping-pong between the two pairs of buffers (§2.4), one direction per pass. Both pass counts are even, so the sorted pairs end in the first pair, where the cull wrote them and where `FrameBuffers::read_back` and the draw read them; no copy is needed.

The sort is a compute pass of its own after the prepare pass, with the frame's timestamps 2 and 3 (Q-81). Every dispatch is indirect (stage 4): the count and scatter kernels read the block count stage 3 wrote, and the scan reads a second argument triple stage 3 writes for it, one workgroup or none when nothing is visible (Q-T113.1). The kernels read the clamped visible count from the counters, so keys left past it by an earlier frame are never counted or moved. Nothing is read back to size the sort (D-34).

## Workgroup layout

Every kernel runs 256 threads per workgroup (D-33). A block is 1024 keys, four per thread (Q-T113.1): with 256 32-bit bins per block the histogram buffer holds about one byte per unit of C, the size §1.12 and Q-23 give it, which a block of 256 keys could not. Stage 3's dispatch arguments are therefore ⌈count / 1024⌉ blocks, the value R-4 and R-6 check through `FrameContents.sort_dispatch`.

The scatter is stable without any cross-thread primitive beyond workgroup memory and barriers. It walks its block in four rounds of 256 consecutive keys, one per thread in key order. In a round each thread sets its own bit in its digit's 256-bit mask in workgroup memory; after a barrier, a key's rank among the round's keys of its digit is the number of bits set below its own, counted with `countOneBits`. Its output position is the scan's offset for (block, digit), plus the keys of that digit in the block's earlier rounds, plus that rank. After the round each thread advances its own digit's running offset by the round's count for that digit and clears the digit's mask. Ranks follow thread order within a round, rounds follow key order within a block, and the scan orders blocks, so equal digits keep their input order and the whole sort is stable.

## Why no subgroup operations

D-33 rules out subgroup (wave) operations and decoupled look-back: the sorts the research surveyed guess the subgroup size or fail on mobile drivers (Appendix A rows 24–29), and the Quest's Adreno driver must run the same kernels as a desktop GPU (Dv-3). The kernels use only workgroup memory, workgroup barriers, workgroup atomics and `countOneBits`, all core WGSL; a unit test validates each shader with no optional capability. The price is a scan by a single workgroup, which grows with the number of blocks; its cost is measured by the benchmarks, not by R-4.

## How R-4 reaches the sort

`SplatPasses` has no public sort entry point (PRD #76). R-4 (`tests/r04_sort.rs`) builds a model whose 10⁶ rows are all visible at random depths, runs `encode_prepare` and reads the frame back with `FrameBuffers::read_back` (Q-69). It recomputes each slot's key from its decoded record's depth with `sort_key` (`sort_key16` for a `Depth16` tier, Q-2) and checks that `sorted` equals the slots stably sorted by that key, for random depths, for depths repeated a thousand times each (stability), and for `Depth16`, together with the block count in `sort_dispatch`.

The `pub(crate)` encoder, `RadixSort`, has private tests in its own module that write keys, payloads, the count and the dispatch arguments directly and run the sort alone: a partial last block, an exact multiple of the block, one key and none, stale keys past the count, all-equal keys, few distinct keys across many blocks, `Depth16`, and 10⁶ random keys.
