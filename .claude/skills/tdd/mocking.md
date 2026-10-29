# When to Mock

Mock only at **system boundaries**, and prefer a real stand-in to a mock. The boundaries and their stand-ins below come from openftgs, until the openftgs-xr spec names this package's own:

| Boundary | What sits there | Use in tests |
| --- | --- | --- |
| GPU rasterizer | gsplat (CUDA) | The pure-PyTorch reference backend (`render(..., backend="reference")`) on CPU. GPU-marked tests compare it with gsplat. Never mock the renderer. |
| Downloaded weights | RoMa, DINOv2, the LPIPS networks | The real weights, in tests carrying the weights marker of `pyproject.toml`; tests without it don't touch the network. |
| Separate programs | ffmpeg, COLMAP, BackgroundMattingV2 | A stub executable that records its input and writes the expected output, as openftgs U-37 does for COLMAP. |
| Randomness | `torch.Generator` | A seeded generator passed through the public signature (`generator=` in §2). Never seed or patch the global generator. |
| Datasets | Neural3DV, ENeRF-Outdoor, SelfCap | Generated mini-fixtures, like openftgs U-37's mini-archive. Real data only in dataset-gated tests, and no dataset material is ever committed (CLAUDE.md, "Commits"). |
| File system | checkpoints, exchange files, manifests | Real files under pytest's `tmp_path`. |

Don't mock:

- Anything inside `openftgsxr`, including vendored code
- Internal collaborators (in openftgs, `mcmc_split` inside `relocate`)
- Anything you control

## Designing for testability

The §2 signatures inject what tests need to control (in openftgs: the generator, the render backend, the configuration objects, the paths). Test through them. Don't add a parameter, hook or global for testability alone: changing a §2 signature means stopping to ask the owner, adding a public signature the spec omits is a Q-n row (CLAUDE.md), and anything private you add for a test is an implementation detail the test then depends on.
