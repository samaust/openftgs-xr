# Fixtures

How `ftgsxr-testkit` meets §2.7's Fixtures paragraph and D-29: what each committed fixture under
`tests/fixtures/` is for, how `tools/fixtures/` regenerates it, how the checksum check gates the
suite, and which test inputs are made at test time instead of committed. The file names are the
contract of the Layout section of `tests/fixtures/README.md`; this doc does not repeat them in
full.

## The checksum gate

`fixtures_dir()` (Q-1) is the only way a test reaches `tests/fixtures/`. The first call in a test
process reads `SHA256SUMS` and hashes every file it lists, in its order; the result is kept for the
process, so the check runs once however many tests ask. While it fails, every call panics with a
message naming the first file that is missing or whose digest differs, so no test runs on a
changed fixture (§2.7: "A fixture's checksum mismatch fails the suite before any test"). A path
in `SHA256SUMS` may hold spaces (the listing's model named with one does).

The check covers the files `SHA256SUMS` lists. A fixture added without rewriting the sums is not
checked, so every change to `tests/fixtures/` rewrites `SHA256SUMS`: `make_all.py` does it after
regenerating, and the README gives the command for hand-written files.

The verification itself is a private helper, pinned by private tests over a scratch directory
and over a copy of the committed fixtures with one byte changed.

## What each fixture serves

| Fixture (README Layout) | Made by | Serves | Rows it relies on |
| --- | --- | --- | --- |
| `probes/P1` … `P10`: probe models, render requests, reference renders | `make_probes.py`, from openftgs's harness probe definitions, then `ftgs render --backend reference` | R-1; the readers' U-1, U-4, U-24, U-25 (an openftgs-written renders file) and U-33 | none |
| `probes/R2_*`: R-2's additional probe models | `make_probes.py` | R-2 | Q-10, Q-11 (the reduced random probe) |
| `probes/R19_transmittance` | `make_probes.py` | R-19 | Q-10 |
| `probes/R2_overlap_deg3.tau1.3.ply`: a PLY snapshot of a probe model | openftgs's `write_ply` | U-6 | Q-10 |
| `probes/R7_merged`, `R8_merged` and `merged/`: the pre-posed merged models, the instances and their poses | `merge_posed.py` | R-7, R-8 | Q-12 |
| `checkpoint/tiny/`, `tiny.safetensors`, `tiny.tau0.ply`: D-29's tiny checkpoint, its exchange model and its PLY snapshot | `make_checkpoint.py`, running `ftgs train` and `ftgs export` on the CPU | U-5, U-33, the PLY reader (D-41) | none |
| `checkpoint/single/`, `single.safetensors`: the same on one frame, a single-timestamp checkpoint | `make_checkpoint.py` | U-5's single-timestamp clause | none |
| `checkpoint/scene/`: the scene manifest of the tiny training scene and its images | `make_checkpoint.py` | U-36; the input of the two training runs | none |
| `listing/`: the three listing bodies and the files they list | written by hand from the documented formats; `make_listings.py` fills in sizes and digests | U-27; U-28 and U-31 through the fixture server | Q-6 |
| `README.md` | `make_all.py`, from its template | provenance: the openftgs commit, the package versions, the commands, the adapter reply | Q-13 |

The sample settings, scene and pose files of every version that §2.7 lists (U-29, U-30) are
hand-written by the tickets that need them, beside the generated files, and enter `SHA256SUMS`
the same way.

## Regenerating

`tools/fixtures/make_all.py` (Python dev tooling, importing nothing from the workspace) runs the
scripts above in order with openftgs as a program on the CPU, writes the README and rewrites
`SHA256SUMS` over the whole directory, hand-written files included. openftgs's output is
deterministic in content but not byte for byte, so the script keeps the previous bytes of every
file whose content did not change and names any checksum that still changed; on the README's
openftgs commit and package versions a rerun changes nothing. Trained models are never committed:
the `ds` lane's models are written under `$OPENFTGS_DATASETS/xr-fixtures/` by M-2's tooling
(D-29).

## Made at test time, not committed

Three testkit helpers build model inputs in memory, so tests that need many or invalid files do
not grow the committed set:

- `random_model(seed, n, degree)`: a random exchange model that passes `Model::validate`, the
  same for the same arguments, with time centers over the span and a few persistent rows (E3)
  among transient ones; it serves U-8 and the readers' round trips. Its `identity` is the
  SHA-256 of its own exchange file, so a reader given `exchange_bytes` of it can return an equal
  model.
- `exchange_bytes(&model)`: a model as a `ftgs-if` file with §1.2's tensors and the metadata
  openftgs's writer records, every tensor aligned in the file.
- `SafetensorsBuilder`: the safetensors container written byte by byte, with any dtype, shape and
  metadata, extra header padding and a data section moved off its alignment, for U-2's invalid
  files and U-3's padded and unaligned containers, which a conforming writer would not produce.

`real_sh` sits beside them as U-9's independent evaluation of E4's basis; it reads no fixture.

## Read through helpers

- `probe(name)` and `reference_render(name)` return a committed probe model with its render
  request, and its reference renders checked against that request's digest, by the README's
  probe names (Q-T15.3). `metrics(a, b)` is E15 with D-46's clamp, U-21's subject; a
  reference render against itself is exact.
- `serve_fixtures()` serves the whole directory over HTTP on the loopback for U-28 and U-31
  (Q-T15.1): each file at its relative path with an ETag and a fixed `Last-Modified` taken from
  the listing bodies, conditional requests answered by 304, a single byte range by 206. The
  `listing/` bodies are served at the listing base `listing/models/`, where their relative paths
  resolve to the listed copies; `serve_listing` picks which of the three answers there (Q-6), so
  one server shows each flavor and the order D-31 detects them in, through its request records.
  `truncate_next` breaks the next body after a given number of bytes, so a download can be
  resumed (Q-T15.2). Each call starts its own server, so tests running in parallel do not share
  records, listing choice or truncation.
