// SPDX-License-Identifier: Apache-2.0

//! Q-436 (#736, Q-325, §2.4 Threads, `gpu`): `SplatPasses::new` compiles on several
//! threads of one device at once, as the adapter's warm-up threads do. Each call's
//! validation scope is its own thread's, so no call panics on another's scope
//! ("Mismatched pop_error_scope call", the panic #722 saw in the warm-up threads) and each
//! returns its passes.

use std::sync::{Arc, Barrier};
use std::thread;

use ftgsxr_render::{QualityTier, SplatPasses};
use ftgsxr_testkit::{gpu_context, layer_messages};

/// The threads compiling at once: the warm-up's two.
const THREADS: usize = 2;
/// The compiles each thread runs, so their scopes overlap in some of them.
const ROUNDS: usize = 5;

#[test]
#[ignore = "gpu"]
fn gpu_splat_passes_compile_at_once_on_one_device_from_two_threads() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = Arc::new(gpu_context());
    let start = Arc::new(Barrier::new(THREADS));
    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let dev = Arc::clone(&dev);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                (0..ROUNDS)
                    .map(|_| {
                        SplatPasses::new(
                            &dev,
                            wgpu::TextureFormat::Rgba16Float,
                            Some(wgpu::TextureFormat::Depth32Float),
                            &QualityTier::conformance(),
                        )
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for (i, t) in threads.into_iter().enumerate() {
        let results = t
            .join()
            .unwrap_or_else(|_| panic!("thread {i}'s compiles do not panic"));
        for (round, r) in results.into_iter().enumerate() {
            assert_eq!(r, Ok(()), "thread {i}, round {round}: the passes compile");
        }
    }
}
