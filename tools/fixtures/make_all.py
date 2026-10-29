# SPDX-License-Identifier: Apache-2.0
"""Regenerate every fixture of tests/fixtures/ that tools/fixtures/ makes (§2.7, D-29).

Usage, from anywhere, with the Python environment where openftgs is installed:

    $OPENFTGS_REPO/.venv/bin/python tools/fixtures/make_all.py

It removes and rewrites probes/, merged/, checkpoint/ and listing/ (make_probes.py,
merge_posed.py, make_checkpoint.py, make_listings.py, in that order), then writes README.md (the
provenance: the openftgs commit, the package versions, the commands, the adapter's `info` reply)
and SHA256SUMS over every file of tests/fixtures/, other tickets' hand-written files included.
It ends by naming the files whose checksum changed against the SHA256SUMS it replaced.
Intermediates go to $FIXTURES_WORK (default: the workspace's gitignored .scratch/fixtures-work).
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fixture_common import FIXTURES, THREADS, WORK, environment, program
from make_checkpoint import CONFIG

HERE = Path(__file__).resolve().parent
GENERATED = ("probes", "merged", "checkpoint", "listing")
SCRIPTS = ("make_probes.py", "merge_posed.py", "make_checkpoint.py")
LISTINGS = "make_listings.py"  # last, after `keep_unchanged`: it copies files of the others
CHECKPOINTS = ("checkpoint/tiny", "checkpoint/single")
# A checkpoint's generator states: Python's `random` state is unseeded, so it differs between
# runs without changing any trained number; the player never reads these files.
GENERATOR_STATES = ("rng.json", "rng.safetensors")
SUMS = FIXTURES / "SHA256SUMS"


def checksums() -> dict[str, str]:
    """SHA-256 of every file under tests/fixtures/ but SHA256SUMS, by POSIX relative path."""
    names = sorted(  # byte order, as `LC_ALL=C sort` gives
        p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob("*") if p.is_file()
    )
    return {
        name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        for name in names
        if name != SUMS.name
    }


def read_sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    sums = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    return sums


def safetensors_content(path: Path) -> tuple[object, bytes]:
    """A safetensors file's header as a JSON value and its data bytes: openftgs's writer (the
    `safetensors` package) orders the header's metadata by a per-process hash, so two writes of
    the same tensors and metadata can differ in their bytes."""
    data = path.read_bytes()
    n = int.from_bytes(data[:8], "little")
    return json.loads(data[8 : 8 + n]), data[8 + n :]


def same_content(a: Path, b: Path) -> bool:
    if a.read_bytes() == b.read_bytes():
        return True
    if a.suffix == ".safetensors":
        return safetensors_content(a) == safetensors_content(b)
    return False


def same_checkpoint(a: Path, b: Path) -> bool:
    """Equal checkpoints but for their generator states and the digests the manifest records."""
    names = sorted(p.name for p in a.iterdir())
    if names != sorted(p.name for p in b.iterdir()):
        return False
    for name in names:
        if name in GENERATOR_STATES:
            continue
        if name == "manifest.json":
            first, second = (json.loads((d / name).read_text()) for d in (a, b))
            first.pop("files"), second.pop("files")
            if first != second:
                return False
        elif not same_content(a / name, b / name):
            return False
    return True


def keep_unchanged(previous: Path) -> list[str]:
    """Put back the previous bytes of every regenerated file whose content did not change, and
    of every checkpoint directory that did not change but for its generator states, so that a
    rerun changes no checksum unless something really changed. Returns what was kept."""
    kept = []
    for name in CHECKPOINTS:
        old, new = previous / name, FIXTURES / name
        if old.is_dir() and new.is_dir() and same_checkpoint(old, new):
            shutil.rmtree(new)
            shutil.copytree(old, new)
            kept.append(name + "/")
    for old in sorted(previous.rglob("*")):
        name = old.relative_to(previous).as_posix()
        new = FIXTURES / name
        inside = any(name.startswith(c + "/") for c in CHECKPOINTS)
        if old.is_file() and not inside and new.is_file() and same_content(old, new):
            if old.read_bytes() != new.read_bytes():
                shutil.copyfile(old, new)
                kept.append(name)
    return kept


def git(*args: str) -> str:
    repo = os.environ.get("OPENFTGS_REPO")
    if not repo:
        return "unknown ($OPENFTGS_REPO unset)"
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def adapter_info() -> str:
    """`ftgs-adapter --backend reference info`: openftgs has no `ftgs info` command."""
    # Q-T3.4: §2.7's "`ftgs info`'s reply" is the adapter's reply for the reference backend.
    done = subprocess.run(
        [program("ftgs-adapter"), "--backend", "reference", "info"],
        env=environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.dumps(json.loads(done.stdout), indent=2)


README = """\
# Test fixtures

Generated by `tools/fixtures/make_all.py` (§2.7 Fixtures, D-29); do not edit the generated files
by hand: rerun the script. `SHA256SUMS` covers every file here (this README included) and
`ftgsxr_testkit::fixtures_dir` checks it before any fixture is used.

Hand-written fixtures (the sample settings, scene and pose files) live here beside the generated
ones and are covered by `SHA256SUMS` too. After adding or changing one without rerunning the
script, rewrite the sums from this directory, in the script's own order, with

```sh
find . -type f ! -name SHA256SUMS -printf '%P\\n' | LC_ALL=C sort \\
  | xargs -d '\\n' sha256sum > SHA256SUMS
```

This README is regenerated by the script: edit its template in `tools/fixtures/make_all.py`.

## Provenance

- openftgs commit `{commit}` (`{describe}`{dirty}), used as a
  program and through its Python API (`openftgs.io` writers, the harness's probe builders);
  package versions: openftgs {openftgs},
  ftgs-harness {harness}, torch {torch}, numpy {numpy}, safetensors {safetensors}, Pillow {pillow};
  Python {python}.
- Every openftgs command ran on the CPU (`CUDA_VISIBLE_DEVICES=` empty, so the reference
  backend) with `OMP_NUM_THREADS={threads}`, from this directory.
- openftgs has no `ftgs info` command: the reply below is `ftgs-adapter --backend reference info`,
  the adapter reply of openftgs §3.3's contract that §1.7 refers to, for the backend that made
  the renders (Q-T3.4).

```json
{info}
```

The scripts write the models, requests, initial points, scene manifests and PLY snapshots
through openftgs's Python API and run these openftgs commands, in order (`$FIXTURES_WORK` holds
the intermediates, which are not committed; `checkpoint/tiny/` and `checkpoint/single/` are
copies of `$FIXTURES_WORK/run_tiny/checkpoint/` and `$FIXTURES_WORK/run_single/checkpoint/`):

```sh
{commands}
```

`$FIXTURES_WORK/config.json`, the `TrainConfig` of both training runs:

```json
{config}
```

Reproducibility: openftgs's output is deterministic in content on the CPU at a fixed thread
count, but not byte for byte: its safetensors writer orders a header's metadata by a per-process
hash, and a checkpoint's `rng.json` and `rng.safetensors` hold Python's unseeded generator state
(which no trained number depends on), so the checkpoint's `manifest.json` digests change too.
`make_all.py` therefore keeps the previous bytes of every file whose content (tensors and
metadata) did not change, and of every checkpoint directory that did not change but for its
generator states, and reports any checksum that still changed: on the same openftgs commit and
package versions, a rerun changes none.

## Layout

- `probes/NAME.safetensors`, `NAME.request.json`, `NAME.renders.safetensors`: a probe model
  (`ftgs-if`, written by openftgs's `io.write_exchange`, producer `openftgs-xr tools/fixtures`),
  its `ftgs-render-request` and openftgs's reference renders of it (`ftgs-renders`, written by
  `ftgs render ... --backend reference`, with the request's SHA-256). NAME is:
  - `P1` … `P10`: openftgs's L1 probes (§3.2 R-1), built by the harness's own probe definitions
    with 32 items each;
  - `R2_overlap_deg3`, `R2_white_background`, `R2_single_timestamp`, `R2_epoch_time`,
    `R2_random`: R-2's additional probes (Q-T3.1; `R2_random` is reduced, Q-T3.2);
  - `R19_transmittance`: R-19's near α = 0.95 over far α = 0.999 (Q-T3.1);
  - `R7_merged`, `R8_merged`: the pre-posed merged models of R-7 and R-8 (Q-T3.3).
- `probes/R2_overlap_deg3.tau1.3.ply`: openftgs's `write_ply` of that probe's per-time arrays at
  τ = 1.3 (U-6).
- `merged/R7_instance_a.safetensors`, `merged/R7_instance_b.safetensors`: P7's two rows as two
  models; `merged/R7_poses.json`, `merged/R8_poses.json`: the instances' poses (R-8's instance is
  `probes/R2_overlap_deg3.safetensors`), with the convention stated in each file (Q-T3.3).
- `checkpoint/scene/scene.json`: the `ftgs-scene` manifest of the synthetic two-camera,
  two-frame scene (world z up, metres; camera `cam1` moves between the frames, a per-image
  override; its frame-1 image is the test split), with its four PNG images under
  `checkpoint/scene/images/`; `checkpoint/scene/single.json`: its frame-0 images alone.
- `checkpoint/tiny/`: the `ftgs-checkpoint` directory of `ftgs train` (20 iterations, reference
  backend, 64 primitives, degree 3) on `scene.json`; `checkpoint/tiny.safetensors`: its
  `ftgs export`; `checkpoint/tiny.tau0.ply`: its `write_ply` snapshot at τ₀.
- `checkpoint/single/`, `checkpoint/single.safetensors`: the same training on `single.json`, a
  single-timestamp checkpoint (τ₀ = τ₁, U-5's unit 1), and its `ftgs export`.
- `listing/index.json`, `listing/nginx.json`, `listing/apache.html`: the three listing bodies of
  U-27 and U-31, **written by hand from the documented formats** (the player's `ftgsxr-index`,
  §1.11; nginx's `autoindex_format json` and Apache's `mod_autoindex` HTML, spec Appendix A row
  35), not captured from a server; `tools/fixtures/make_listings.py` fills in sizes and digests.
  They list `listing/models/`: `probe p1.safetensors` (a copy of `probes/P1.safetensors`),
  `snapshot.ply` (a copy of the probe PLY), `tiny/` (copies of the checkpoint's `manifest.json`
  and `params.safetensors`; its index size is their sum) and `notes.txt` (not a model). The nginx
  and Apache bodies also list `.hidden.safetensors`, which does not exist; the Apache body has
  the parent link `/listing/` and the column-sort links.

No trained dataset model is committed (§2.7): `train_tiny.sh` (M-2) writes those under
`$OPENFTGS_DATASETS/xr-fixtures/`.
"""


def write_readme(commands: list[str]) -> None:
    status = git("status", "--porcelain")
    text = README.format(
        commit=git("rev-parse", "HEAD"),
        describe=git("describe", "--tags", "--always"),
        dirty=", with local changes" if status and status != "unknown" else "",
        openftgs=version("openftgs"),
        harness=version("ftgs-harness"),
        torch=version("torch"),
        numpy=version("numpy"),
        safetensors=version("safetensors"),
        pillow=version("pillow"),
        python=sys.version.split()[0],
        threads=THREADS,
        info=adapter_info(),
        commands="\n".join(commands),
        config=json.dumps(CONFIG, indent=2),
    )
    (FIXTURES / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    before = read_sums()
    shutil.rmtree(WORK, ignore_errors=True)
    previous = WORK / "previous"
    previous.mkdir(parents=True)
    for name in GENERATED:
        if (FIXTURES / name).exists():
            shutil.move(FIXTURES / name, previous / name)
    for script in SCRIPTS:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], env=environment(), check=True)
    kept = keep_unchanged(previous)
    print(f"kept the previous bytes of {len(kept)} files whose content did not change")
    print(f"== {LISTINGS}", flush=True)
    subprocess.run([sys.executable, str(HERE / LISTINGS)], env=environment(), check=True)
    log = WORK / "commands.jsonl"
    commands = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    write_readme(commands)
    sums = checksums()
    SUMS.write_text("".join(f"{d}  {n}\n" for n, d in sums.items()), encoding="utf-8")
    changed = sorted(n for n in sums.keys() | before.keys() if sums.get(n) != before.get(n))
    if before and changed:
        print(f"{len(changed)} checksums changed:\n  " + "\n  ".join(changed))
    elif before:
        print("no checksum changed")
    total = sum((FIXTURES / n).stat().st_size for n in sums)
    print(f"{len(sums)} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
