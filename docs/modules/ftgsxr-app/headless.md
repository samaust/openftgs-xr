# The headless commands in `ftgsxr-app`

How `ftgsxr_app::headless` meets §1.7's command line and §2.1's signatures: the commands, their exit codes, the error they return, and the Q-n rows the module relies on. The spec items are §1.7 (command line), §1.15, §2.1 `ftgsxr_app::headless`, D-36 and U-33, U-34, R-15; this doc explains them and never overrides them. Begun in M-2 (#117) with the commands that need no GPU; `cmd_render` joined with #118.

## The commands

| Command | Function | Success | Failure |
| --- | --- | --- | --- |
| `ftgsxr render MODEL REQUEST.json OUT.safetensors [--tier TIER]` | `cmd_render(model, request, out, tier, version)` | writes `OUT` as `ftgs-renders`, prints nothing, exit 0 | a request the model refuses or no GPU, exit 2; a non-finite render, exit 3; nothing written |
| `ftgsxr info [--json]` | `cmd_info(version, json)` | prints the adapter reply, exit 0 | none |
| `ftgsxr validate MODEL ...` | `cmd_validate(models)` | one line of header information per model, exit 0 | the first invalid model, exit 2 |
| `ftgsxr import IF OUT` | `cmd_import(input, out)` | writes `OUT`, prints nothing, exit 0 | exit 2, nothing written |
| `ftgsxr export MODEL OUT` | `cmd_export(model, out)` | as `import` | as `import` |

A usage error (a missing argument, an unknown subcommand) is clap's, and exits 2 too. The binary prints a failure as one line `ftgsxr: <code>: <user text>` on standard error and exits with `exit_code` (Q-77).

- **`render`.** The checks that need no GPU come first, so a wrong request is reported as such on any machine: the request is read (`read_request`, openftgs's checks), then the model (`open_model`, any §1.2 input), then the request's `time_unit` against the model's (E-REQ-UNIT) and its `sh_degree` against the model's L (E-REQ-DEGREE; null is L). Only then does the command open a `HeadlessRenderer` in the tier, which makes its own wgpu device without a Bevy `App` (§2.1 Builds), and load the model at the request's degree (Q-86). Each item renders at its timestamp from its camera, at the identity pose, over the request's background, rgb unclamped. The renders are kept in memory and the file is written once, after the last item, through `write_renders` (a temporary sibling renamed over `OUT`), so a failure on any item writes no file and leaves an older `OUT` as it was. The file's `producer` is `openftgs-xr <version> (<GPU adapter name>)` and its `request_sha256` the SHA-256 of the request file's bytes (Q-T118.2). The tier defaults to `conformance`; `--tier` takes the five names of §1.12 and clap refuses any other.
- **`info`.** The reply is built once (name, version, no query kinds, the rasterizer, openftgs's conventions with the transmittance stop null, D-7) and written two ways: with `--json` as `info_json`'s compact object, the form openftgs's harness reads; without it as one `key: value` line per field in the same order, the conventions flattened to `render_conventions.<key>` (Q-71), strings bare and other values as their JSON text (Q-77).
- **`validate`.** Each model is read in full and validated (`open_model`), not only its header, so every reader rule of §1.2 applies. The `ModelInfo` it returns comes from the loaded model; its size is that of the file the model's identity hashes, `params.safetensors` for a checkpoint (D-40, Q-76). The signature returns one `Result`, so when a model fails, the binary prints no line for the models before it.
- **`import` and `export`.** §1.7 gives the two commands one rule, so they share one private function. The input is validated first; then a `ftgs-if` input is copied byte for byte (the bytes copied are checked against the identity the validated read computed, so a file replaced between the two reads is refused), and a checkpoint or a PLY is written in its activated form through the core's `write_exchange`, which names the producer `openftgs-xr <version>` (Q-65, Q-72). Both writes go through a temporary sibling renamed over `OUT`, so a failed write leaves the old file, or none, and no temporary.

## `HeadlessError` and the exit codes

`Input` and `Environment` exit 2, `Numeric` exits 3 (§2.1, §1.15). §2.1's variants carry only a message, so the §1.15 code travels in the message's leading `E-…:` prefix (Q-64): `code()` reads it back, and `user_text()` is the rest. A message without a recognized prefix still has a code, E-FILE-FORMAT for `Input` and E-GPU-ADAPTER for `Environment` (Q-75). The commands turn every `CoreError` they meet, whether reading the input or writing the output, into `Input("<code>: <path>: <core user text>")`, so the line the user sees names the file that failed (Q-75).

`render` adds the render layer's errors (Q-T118.2). E-REQ-DEGREE has no `CoreError`, so its message is composed with the prefix directly: `Input("E-REQ-DEGREE: <request>: Request degree X, model degree Y")`. A `RenderError` becomes `Environment("<code>: <user text>")` for the GPU conditions (no adapter, a lost device after its one recovery, a shader that fails, a model over the tier's memory budget), all exit 2. A non-finite render becomes `Numeric("item <id>")`, exit 3, and a core error inside the render layer becomes Q-75's `Input` on the model's path.

## Dependencies

The module takes `ftgsxr-core` and, for `render`, `ftgsxr-render`'s `HeadlessRenderer` (§2.1's band table), which brings Bevy's render crates into the build. The binary depends on `ftgsxr-app` and clap alone (§2.1). So it cannot name the core's `ModelKind`, and it tells a model's kind by its variant's name when printing `validate`'s lines. `ftgsxr_app` re-exports `QualityTier`, the type `cmd_render` takes, so the binary can turn `--tier`'s name into a tier with `QualityTier::by_name` (Q-T118.1).
