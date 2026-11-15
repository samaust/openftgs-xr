# The radix sort in `ftgsxr-render`

How `ftgsxr_render::gpu` sorts a view's records into draw order: stage 4 of §1.4's per-frame pipeline, the player's own radix sort of D-33, run inside `SplatPasses::encode_prepare`. The spec items are §1.4 stages 3 and 4, E8, D-33, D-34 and §2.4's GPU data layout, with R-4 as their test; this doc explains them and never overrides them.

## What is sorted

The cull pass (stage 2) gives each visible splat record a slot and writes, at that slot, the record's sort key k̄ (E8, always the full 32 bits, also in a `Depth16` tier) into the first key buffer and the slot itself into the first payload buffer. Stage 3 clamps the visible count to the capacity C and writes the sort's dispatch arguments. Stage 4 then sorts the (k̄, slot) pairs by ascending k̄, far to near, stably, so records of equal key keep their slot order (E8). The splat draw (stage 5) reads the sorted slots from the first payload buffer. Center records are never sorted: they follow the splats in the record buffer (Q-78) and stage 4 reads only the splat count.

## The passes

A least-significant-digit radix sort with 8-bit digits: four passes for `Depth32`, from the lowest digit up, and two for `Depth16`, over the top two digits only, which is E8's 16-bit key without a second key buffer. Each pass runs three kernels:

- **Count** (`sort_count.wgsl`). Each workgroup takes one block of consecutive keys, counts the pass's digit in a histogram in workgroup memory, and writes all 256 bins of its block (zeros included, so nothing needs clearing) to the histogram buffer.
- **Scan** (`sort_scan.wgsl`). One workgroup turns the blocks' histograms in place into scatter offsets, in one walk (§2.4's single-workgroup scan): thread d walks digit d's column from block 1 on, leaving at each block the keys of digit d in the blocks before it; the 256 digit totals are scanned in workgroup memory; and block 0's row, whose column prefix is always 0, receives each digit's base, the keys of all smaller digits (#123).
- **Scatter** (`sort_scatter.wgsl`). Each workgroup takes the same block again and moves every key and its payload to its output position in the other buffer pair: the digit's base from block 0's row, plus its block's column prefix (none for block 0), plus the key's rank among the block's keys of its digit.

Keys and payloads ping-pong between the two pairs of buffers (§2.4), one direction per pass. Both pass counts are even, so the sorted pairs end in the first pair, where the cull wrote them and where `FrameBuffers::read_back` and the draw read them; no copy is needed.

The sort is a compute pass of its own after the prepare pass, with the frame's timestamps 2 and 3 (Q-81). Every dispatch is indirect (stage 4): the count and scatter kernels read the block count stage 3 wrote, and the scan reads a second argument triple stage 3 writes for it, one workgroup or none when nothing is visible (Q-85). The kernels read the clamped visible count from the counters, so keys left past it by an earlier frame are never counted or moved. Nothing is read back to size the sort (D-34).

## Workgroup layout

Every kernel runs 256 threads per workgroup (D-33). A block is 1024 keys, four per thread (Q-85): with 256 32-bit bins per block the histogram buffer holds about one byte per unit of C, the size §1.12 and Q-23 give it, which a block of 256 keys could not. Stage 3's dispatch arguments are therefore ⌈count / 1024⌉ blocks, the value R-4 and R-6 check through `FrameContents.sort_dispatch`.

The scatter is stable without atomics or any cross-thread primitive beyond workgroup memory and one barrier (#123). Thread t holds the block's keys 4t to 4t + 3, so the block must stay four keys per thread, and writes their four 8-bit digits as one word of a 256-word array in workgroup memory. After the barrier, a key's rank among the block's keys of its digit is the number of equal digits before it: in its own word's earlier bytes, and in the words of threads 0 to t − 1, which it reads one word at a time and compares four bytes at once (an exact zero-byte count with `countOneBits`). The threads of a wave read the same word together, so the reads are broadcasts. Ranks follow key order within the block and the scan orders the blocks, so equal digits keep their input order and the whole sort is stable. Keys past the visible count are never read: they sit after every live key of the last block, so no rank counts them.

## Where the time goes

B-3 (§3.5) times the sort; its split by kernel was measured for #123 with probe builds that dispatch only some kernels, never merged. At C = 10⁶ before #123, the scan took most of the RTX 4090's time (about 1.2 of 1.46 ms): one workgroup walking a column one block after another waits on the memory's latency at every step, so the walk now loads 16 blocks' entries before adding them, and walks once instead of twice. On the Quest 3 (`quest-balanced`) the scatter took most (about 17 of 21 ms), nearly all of it in ranking with workgroup atomics on 256-bit masks over four rounds of four barriers, and little in its scattered writes, which a probe writing in place barely changed. The atomic-free ranking above took it to about 14 ms. A larger block doesn't help this ranking, since each thread's loop grows with the keys it holds; a table of per-chunk counts that shortens the loop measured slower on both machines.

A probe that keeps the count and the scan but replaces the scatter with a plain copy of keys and payloads takes about 4.7 ms on the Quest at C = 10⁶: four passes of D-33's classic sort cannot get under that there, above P-4's Quest limit (§1.12). The figures are on #123.

## Why no subgroup operations

D-33 rules out subgroup (wave) operations and decoupled look-back: the sorts the research surveyed guess the subgroup size or fail on mobile drivers (Appendix A rows 24–29), and the Quest's Adreno driver must run the same kernels as a desktop GPU (Dv-3). The kernels use only workgroup memory, workgroup barriers, workgroup atomics (in the count) and `countOneBits`, all core WGSL; a unit test validates each shader with no optional capability. The price is a scan by a single workgroup, which grows with the number of blocks, and a scatter that ranks by counting instead of with a subgroup's ballot; their cost is measured by the benchmarks ("Where the time goes"), not by R-4.

## How R-4 reaches the sort

`SplatPasses` has no public sort entry point (PRD #76). R-4 (`tests/r04_sort.rs`) builds a model whose 10⁶ rows are all visible at random depths, runs `encode_prepare` and reads the frame back with `FrameBuffers::read_back` (Q-69). It recomputes each slot's key from its decoded record's depth with `sort_key` (`sort_key16` for a `Depth16` tier, Q-2) and checks that `sorted` equals the slots stably sorted by that key, for random depths, for depths repeated a thousand times each (stability), and for `Depth16`, together with the block count in `sort_dispatch`.

The `pub(crate)` encoder, `RadixSort`, has private tests in its own module that write keys, payloads, the count and the dispatch arguments directly and run the sort alone: a partial last block, an exact multiple of the block, one key and none, stale keys past the count, all-equal keys, few distinct keys across many blocks, `Depth16`, and 10⁶ random keys.
