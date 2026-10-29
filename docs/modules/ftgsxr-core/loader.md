# The loader in `ftgsxr-core`

How `ftgsxr_core::loader` turns the three files of §1.2 (an exchange model, a checkpoint, a PLY snapshot) into one validated `Model`: the validation order the three readers share and why, the safetensors container's cast-or-copy path, where the model identity is computed, and what a header-only read can and cannot tell. The spec items are §1.2, §1.15's E-FILE codes, D-8, D-40 and D-41, with openftgs §2.2; this doc explains them and never overrides them. The provisional decisions it relies on are Q-8 and Q-T12.1 to Q-T12.6 in `docs/prd/questions.md`.

## One shape for three readers

Each reader does the same three things in the same order:

1. **The file's own checks**, which only that reader can make: the safetensors container and the `ftgs-if` header for an exchange model; `manifest.json`, the parameters file's digest and its nine stored tensors for a checkpoint; the PLY header and body for a snapshot.
2. **Building the `Model`** in the exchange form: copied as it is from an exchange model, activated from a checkpoint's stored tensors (§1.2's formulas, with openftgs's precision per field), reassembled from a snapshot's properties as a single-timestamp model.
3. **`Model::validate`**, the reader rules that hold for any model in memory (§1.2), then the normalization of every rotation with the canonical sign.

So the rules a `Model` can break are written once, in `types`, and every file kind is held to them; the readers only add what a `Model` no longer shows: the format, the version, the keys, the dtypes and the shapes.

## The validation order

A file with several faults is reported with the first one in §1.2's order (PRD #1, "Reader internals"): format, version, the metadata keys and numbers, the tensors present, their dtypes and shapes, then `Model::validate`'s rules on the values (finite, positive durations and scales, rotation norms, the opacity range, the single-timestamp rule), and the `render_conventions` block last. The order runs from what makes the file unreadable to what makes one value wrong, so the message names the deepest fault the reader could see. The conventions block comes last although it is metadata: a block that does not parse is held back while the arrays are validated with openftgs's defaults, and reported only if they pass (Q-8, Q-T12.2).

Each fault carries one §1.15 code, mapped as `formats` maps its own files (Q-T12.2): a container that is not safetensors, JSON or PLY, or another `format`, is E-FILE-FORMAT; a missing key E-FILE-KEYS; a version not of major 1 E-FILE-VERSION; a dtype or shape E-FILE-SHAPE, except an SH coefficient count that disagrees with the degree, which is E-FILE-SH; everything else on a value E-FILE-VALUE or E-FILE-SINGLE. A checkpoint's digest is checked before any fault inside its parameters file, so a corrupted file is reported as E-FILE-CHECKSUM rather than as whichever tensor it garbled (Q-T12.5). A PLY snapshot follows the same mapping (Q-T12.6).

Rotations are checked raw: a quaternion's norm must exceed 10⁻⁸ in the file, and only then is it normalized. A writer's already-normalized quaternion is kept bit for bit, so an exchange model reads back exactly as it was written (Q-T12.3).

## The container's cast-or-copy path

An exchange model and a checkpoint's parameters are safetensors containers (§1.2): the 8-byte header length, the JSON header beginning with `{` (and possibly padded with spaces), then the tensor data at offsets relative to the data section. Nothing aligns that data: the header's length decides where it starts. Each tensor is therefore read one of two ways into the `Model`'s owned arrays: when its bytes happen to sit at an address aligned for the element type (and the machine is little-endian), the slice is reinterpreted in place and copied in one move; otherwise each value is assembled from its little-endian bytes. Both paths give the same values; U-3 pins it by reading one model shifted by every offset from 0 to 7.

The `safetensors` crate parses the header and checks that the offsets cover the data section without holes, each with the size its dtype and shape imply; the loader adds the `{` rule and the 100 MB header limit's message, and maps every container fault to E-FILE-FORMAT.

`open_model` reads the file into memory and passes the bytes to the reader its kind selects (a directory holding `manifest.json`, a `.ply` suffix, anything else). §2.1 describes that step as a memory map, which `ftgsxr-core` cannot make without `unsafe` (§2.6); the readers take a byte slice, so a mapping can replace the read without touching them once the owner settles Q-T12.1.

## Where the identity is computed

The model identity (D-40) is the SHA-256 of the bytes the reader was given, computed on one thread while the other decodes and validates them (`rayon::join`), so the file is read once by each and the digest costs no extra pass over memory that has gone cold. For a checkpoint the bytes are those of `params.safetensors`, which also lets the reader check the manifest's `files` entry against the same digest. The identity is set only on a model that passed every rule; a refused file has none. Above a fixed row count the activations of a checkpoint and the reassembly of a snapshot run in parallel over rows; `Model::validate` itself runs sequentially.

## What a header-only read tells

`read_info` serves §1.5's library: it reads the JSON header alone (the bytes after the 8-byte prefix, which a range request of 8 + header-length bytes fetches) and the file's size (Q-T12.4). From them it knows everything the header states and checks it as `read_exchange` would, in the same order: the container's offsets against the size, the format and version, the metadata keys and numbers (N, L, τ₀, τ₁, the unit, the producer, the conventions block), and every tensor's presence, dtype and shape. What it cannot know is any value: a non-finite position, a non-positive scale, an opacity of 1 or a moving primitive in a single-timestamp model pass the header and are refused only when the file is read whole. The identity is not known either, since it is the digest of the whole file. Only an exchange model has a header-only read; a checkpoint or a snapshot is listed without that information (Q-T12.4).
