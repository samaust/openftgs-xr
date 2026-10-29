# When to Mock

Mock only at **system boundaries**, and prefer a real stand-in to a mock. The boundaries and their stand-ins below come from openftgs, until the openftgs-xr spec names this package's own:

| Boundary | What sits there | Use in tests |
| --- | --- | --- |
| GPU rasterizer | gsplat (CUDA) | The pure-PyTorch reference backend (`render(..., backend="reference")`) on CPU. openftgs's GPU tests compare it with gsplat; openftgs-xr's `gpu` lane compares its renders with committed reference renders. Never mock the renderer. |
| Downloaded weights or files | RoMa, DINOv2, the LPIPS networks (openftgs); model files on an HTTP server (openftgs-xr) | The real downloads, in the `network` lane; tests outside it serve fixtures from a local test server or from files. |
| Separate programs | ffmpeg, COLMAP, BackgroundMattingV2 | A stub executable that records its input and writes the expected output, as openftgs U-37 does for COLMAP. |
| Randomness | `torch.Generator` (openftgs); a `rand` RNG (openftgs-xr) | A seeded generator passed through the public signature. Never seed or patch a global one. |
| Datasets | Neural3DV, ENeRF-Outdoor, SelfCap | Generated mini-fixtures, like openftgs U-37's mini-archive. Real data only in the `dataset` lane, and no dataset material is ever committed (CLAUDE.md, "Commits"). |
| File system | checkpoints, exchange files, manifests | Real files under a `tempfile::TempDir`. |

Don't mock:

- Anything inside this workspace's crates, including vendored code
- Internal collaborators (in openftgs, `mcmc_split` inside `relocate`)
- Anything you control

## Designing for testability

The §2 signatures inject what tests need to control (in openftgs: the generator, the render backend, the configuration objects, the paths). Test through them. Don't add a parameter, hook or global for testability alone: changing a §2 signature means stopping to ask the owner, adding a public signature the spec omits is a Q-n row (CLAUDE.md), and anything private you add for a test is an implementation detail the test then depends on.
