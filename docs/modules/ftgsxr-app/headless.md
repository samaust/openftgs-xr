# The headless commands in `ftgsxr-app`

How `ftgsxr_app::headless` meets §1.7's command line and §2.1's signatures: the commands, their exit codes, the error they return, and the Q-n rows the module relies on. The spec items are §1.7 (command line), §1.15, §2.1 `ftgsxr_app::headless`, D-36 and U-33, U-34; this doc explains them and never overrides them. Begun in M-2 (#117) with the commands that need no GPU; `cmd_render` joins with #118.

## The commands

| Command | Function | Success | Failure |
| --- | --- | --- | --- |
| `ftgsxr info [--json]` | `cmd_info(version, json)` | prints the adapter reply, exit 0 | none |
| `ftgsxr validate MODEL ...` | `cmd_validate(models)` | one line of header information per model, exit 0 | the first invalid model, exit 2 |
| `ftgsxr import IF OUT` | `cmd_import(input, out)` | writes `OUT`, prints nothing, exit 0 | exit 2, nothing written |
| `ftgsxr export MODEL OUT` | `cmd_export(model, out)` | as `import` | as `import` |

A usage error (a missing argument, an unknown subcommand) is clap's, and exits 2 too. The binary prints a failure as one line `ftgsxr: <code>: <user text>` on standard error and exits with `exit_code` (Q-77).

- **`info`.** The reply is built once (name, version, no query kinds, the rasterizer, openftgs's conventions with the transmittance stop null, D-7) and written two ways: with `--json` as `info_json`'s compact object, the form openftgs's harness reads; without it as one `key: value` line per field in the same order, the conventions flattened to `render_conventions.<key>` (Q-71), strings bare and other values as their JSON text (Q-77).
- **`validate`.** Each model is read in full and validated (`open_model`), not only its header, so every reader rule of §1.2 applies. The `ModelInfo` it returns comes from the loaded model; its size is that of the file the model's identity hashes, `params.safetensors` for a checkpoint (D-40, Q-76). The signature returns one `Result`, so when a model fails, the binary prints no line for the models before it.
- **`import` and `export`.** §1.7 gives the two commands one rule, so they share one private function. The input is validated first; then a `ftgs-if` input is copied byte for byte (the bytes copied are checked against the identity the validated read computed, so a file replaced between the two reads is refused), and a checkpoint or a PLY is written in its activated form through the core's `write_exchange`, which names the producer `openftgs-xr <version>` (Q-65, Q-72). Both writes go through a temporary sibling renamed over `OUT`, so a failed write leaves the old file, or none, and no temporary.

## `HeadlessError` and the exit codes

`Input` and `Environment` exit 2, `Numeric` exits 3 (§2.1, §1.15). §2.1's variants carry only a message, so the §1.15 code travels in the message's leading `E-…:` prefix (Q-64): `code()` reads it back, and `user_text()` is the rest. A message without a recognized prefix still has a code, E-FILE-FORMAT for `Input` and E-GPU-ADAPTER for `Environment` (Q-75). The commands turn every `CoreError` they meet, whether reading the input or writing the output, into `Input("<code>: <path>: <core user text>")`, so the line the user sees names the file that failed (Q-75).

## Dependencies

The commands without a GPU need `ftgsxr-core` only, so the crate's build carries no Bevy for them. The binary depends on `ftgsxr-app` and clap alone (§2.1), so it cannot name the core's `ModelKind`; it tells a model's kind by its variant's name when printing `validate`'s lines.
