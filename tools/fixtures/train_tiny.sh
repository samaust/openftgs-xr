#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The trained models of the `ds` lane (§2.7 Fixtures, D-29, §3.2 R-3): trains openftgs-synth's
# scenes with openftgs, run as a program from its venv, and writes into
# $OPENFTGS_DATASETS/xr-fixtures/ (never committed). Each argument names what to make:
#
#   tiny        (the default) for NAME in tiny, tiny_unsync:
#               NAME.safetensors          the trained model's `ftgs export` (ftgs-if)
#               NAME.request.json         R-3's request, 20 items (tools/fixtures/r3_request.py)
#               NAME.renders.safetensors  `ftgs render ... --backend reference` of it, with the
#                                         request's SHA-256
#   kitchen_1m  the budget model (Q-410, Q-108): openftgs-synth's `kitchen`, 300 frames, at
#               degree 2 and at most 1,000,000 primitives, as kitchen_1m.safetensors; no request
#               and no reference renders. The training holds the GPU for about an hour
#               (openftgs §2.4), so it runs only when asked.
#
# README.md holds the provenance: openftgs commit, versions, commands, config. The budget
# model's provenance is a delimited section of its own; a run rewrites only the parts it makes,
# so a `kitchen_1m` run leaves the tiny models' files and README sections as they are, and a
# `tiny` run keeps the budget model's section. #622's kitchen.safetensors is never touched.
#
# Everything else (the converted frames, the frame cache, the initial points, the run
# directory) goes to a work directory that is removed at the end: only the exports, requests and
# renders are kept.
#
# Usage: OPENFTGS_REPO=... OPENFTGS_DATASETS=... tools/fixtures/train_tiny.sh [tiny] [kitchen_1m]
# Environment: THREADS (CPU threads of every command, default 3), XR_FIXTURES (the output
# directory, default $OPENFTGS_DATASETS/xr-fixtures). `ftgs init roma` and `ftgs train` need a
# CUDA device; the reference renders run on the CPU.
set -euo pipefail

: "${OPENFTGS_REPO:?set OPENFTGS_REPO to the openftgs clone}"
: "${OPENFTGS_DATASETS:?set OPENFTGS_DATASETS to the datasets directory}"
SYNTH=$OPENFTGS_DATASETS/synth
OUT=${XR_FIXTURES:-$OPENFTGS_DATASETS/xr-fixtures}
THREADS=${THREADS:-3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FTGS=$OPENFTGS_REPO/.venv/bin/ftgs
PYTHON=$OPENFTGS_REPO/.venv/bin/python
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

# What to make: the tiny models unless the arguments say otherwise (Q-410: the budget model
# only when asked).
MAKE_TINY=0 MAKE_BUDGET=0
(($# > 0)) || set -- tiny
for ARG in "$@"; do
  case $ARG in
    tiny) MAKE_TINY=1 ;;
    kitchen_1m) MAKE_BUDGET=1 ;;
    *)
      echo "usage: $0 [tiny] [kitchen_1m]" >&2
      exit 2
      ;;
  esac
done

[[ -x $FTGS ]] || { echo "$FTGS not found: install openftgs's venv" >&2; exit 2; }
[[ -f $SYNTH/DATASET_CARD.md ]] || { echo "$SYNTH is not openftgs-synth" >&2; exit 2; }

# Q-84: the tiny models' training configuration, defaults but for the primitive cap (so the
# CPU reference renders stay short) and deterministic kernels.
CONFIG='{
  "growth": {
    "n_max": 100000
  },
  "perf": {
    "deterministic": true
  }
}'
# Q-410: the budget model's, P-1's "1 M primitives at degree 2" with the tiny runs'
# deterministic kernels, defaults otherwise (300 frames at the default 1,000 initial points
# per frame is 300,000, under the cap, as openftgs requires).
BUDGET_CONFIG='{
  "sh_degree": 2,
  "growth": {
    "n_max": 1000000
  },
  "perf": {
    "deterministic": true
  }
}'
SEED=0
# The delimiters of the budget model's README section.
BEGIN='<!-- kitchen_1m: begin (tools/fixtures/train_tiny.sh kitchen_1m) -->'
END='<!-- kitchen_1m: end -->'

mkdir -p "$OUT"
WORK=$(mktemp -d "$OUT/.work.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
COMMANDS=$WORK/commands.txt
: >"$COMMANDS"

# Run a command, recorded in $COMMANDS as the README shows it ($WORK and $OUT spelled as
# variables).
run() {
  local shown="$*"
  shown=${shown//$WORK/\$WORK}
  shown=${shown//$OUT/\$XR_FIXTURES}
  shown=${shown//$SYNTH/\$SYNTH}
  shown=${shown//$HERE/tools/fixtures}
  shown=${shown//$FTGS/ftgs}
  shown=${shown//$PYTHON/python}
  echo "$shown" >>"$COMMANDS"
  echo "+ $shown" >&2
  "$@"
}

# The provenance both kinds of section share.
provenance() {
  VERSIONS=$("$PYTHON" -I -c '
import importlib.metadata as m, sys
names = ["openftgs", "ftgs-harness", "torch", "gsplat", "numpy", "safetensors", "pillow"]
print(", ".join(f"{n} {m.version(n)}" for n in names) + f"; Python {sys.version.split()[0]}")')
  DIRTY=$(git -C "$OPENFTGS_REPO" status --porcelain)
  COMMIT=$(git -C "$OPENFTGS_REPO" rev-parse HEAD)
  DESCRIBE=$(git -C "$OPENFTGS_REPO" describe --tags --always)
  SYNTH_VERSION=$(cat "$SYNTH/VERSION" 2>/dev/null || echo unknown)
  GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo none)
}

# Write README.md from the part a run made: `tiny` replaces everything but the budget
# model's section, which it keeps at the end; `kitchen_1m` replaces or appends that section
# alone. The new file is moved into place whole.
write_readme() {
  "$PYTHON" -I - "$1" "$OUT/README.md" "$2" "$BEGIN" "$END" <<'EOF'
import os, re, sys
mode, readme, part_path, begin, end = sys.argv[1:]
old = open(readme, encoding="utf-8").read() if os.path.exists(readme) else ""
part = open(part_path, encoding="utf-8").read()
found = re.search("\n?" + re.escape(begin) + ".*?" + re.escape(end) + "\n", old, re.S)
block = found.group(0).lstrip("\n") if found else ""
rest = old[: found.start()] + old[found.end() :] if found else old
if mode == "tiny":
    new = part + ("\n" + block if block else "")
else:
    if not rest:
        rest = "# openftgs-xr's trained fixtures (§2.7, D-29)\n"
    new = rest + "\n" + part
with open(readme + ".new", "w", encoding="utf-8") as f:
    f.write(new)
os.replace(readme + ".new", readme)
EOF
}

# Q-84: the conversions. `tiny`: the n3dv form (pinhole frames; cam00 is the converter's
# test view). `tiny_unsync`: the selfcap form, its 12 frames, the dataset card's test camera
# 0002, openftgs's own undistortion (the validation runs' choice).
convert() {
  case $1 in
    tiny)
      run "$FTGS" convert n3dv "$SYNTH/n3dv/tiny" "$WORK/tiny/scene" --frames 12 --resize 1.0 \
        --jobs "$THREADS"
      ;;
    tiny_unsync)
      run "$FTGS" convert selfcap "$SYNTH/selfcap" "$WORK/tiny_unsync/scene" --scene tiny_unsync \
        --window 0:12 --test-camera 0002 --undistort internal --resize 1.0 --jobs "$THREADS"
      ;;
  esac
}

make_tiny() {
  : >"$COMMANDS"
  printf '%s\n' "$CONFIG" >"$WORK/config.json"
  for NAME in tiny tiny_unsync; do
    mkdir -p "$WORK/$NAME"
    convert "$NAME" >/dev/null
    SCENE=$WORK/$NAME/scene/scene.json
    run "$FTGS" init roma "$SCENE" "$WORK/$NAME/init.safetensors" --seed "$SEED"
    run "$FTGS" train "$SCENE" --init "$WORK/$NAME/init.safetensors" --out "$WORK/$NAME/run" \
      --config "$WORK/config.json" --seed "$SEED"
    run "$FTGS" export "$WORK/$NAME/run/checkpoint" --format ftgs-if --out "$OUT/$NAME.safetensors"
    run "$PYTHON" "$HERE/r3_request.py" "$SCENE" "$OUT/$NAME.safetensors" "$OUT/$NAME.request.json"
    run env CUDA_VISIBLE_DEVICES= "$FTGS" render "$OUT/$NAME.safetensors" \
      --request "$OUT/$NAME.request.json" --out "$OUT/$NAME.renders.safetensors" --backend reference
    cp "$WORK/$NAME/run/METRICS.json" "$WORK/$NAME.METRICS.json" 2>/dev/null || true
    rm -rf "${WORK:?}/$NAME"
  done

  provenance
  {
    cat <<EOF
# openftgs-xr's trained fixtures (§2.7, D-29)

Written by openftgs-xr's \`tools/fixtures/train_tiny.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ); not
part of openftgs-synth and never committed. The \`ds\` lane's R-3 reads, for NAME in \`tiny\` and
\`tiny_unsync\`: \`NAME.safetensors\` (the trained model's \`ftgs export\`), \`NAME.request.json\` (R-3's
20 items, Q-83) and \`NAME.renders.safetensors\` (openftgs's reference renders, carrying the
request's SHA-256).

## Provenance

- openftgs commit \`$COMMIT\` (\`$DESCRIBE\`${DIRTY:+, with local changes}), run as a program from its venv.
- Versions: $VERSIONS.
- openftgs-synth $SYNTH_VERSION at \`\$SYNTH\` (\`$SYNTH\`).
- GPU: $GPU (\`init\`, \`train\`); the
  reference renders ran on the CPU (\`CUDA_VISIBLE_DEVICES=\` empty). Every command ran with
  \`OMP_NUM_THREADS=$THREADS\`. Training on the GPU is not bitwise reproducible: a rerun gives other
  models, requests (their τ₀ and τ₁ come from the model) and renders.
- \`\$WORK\`, the intermediates (removed): the converted scenes, frame caches, initial points and
  run directories.

Commands, in order:

\`\`\`sh
EOF
    cat "$COMMANDS"
    cat <<EOF
\`\`\`

\`\$WORK/config.json\`, the \`TrainConfig\` of both runs (TrainConfig's defaults otherwise):

\`\`\`json
$CONFIG
\`\`\`

## Test metrics of the training runs

EOF
    for NAME in tiny tiny_unsync; do
      echo "\`$NAME\` (\`METRICS.json\` of \`ftgs train\`):"
      echo
      echo '```json'
      cat "$WORK/$NAME.METRICS.json" 2>/dev/null || echo "none"
      echo '```'
      echo
    done
    echo "## SHA-256"
    echo
    echo '```'
    (cd "$OUT" && sha256sum tiny.safetensors tiny.request.json tiny.renders.safetensors \
      tiny_unsync.safetensors tiny_unsync.request.json tiny_unsync.renders.safetensors)
    echo '```'
  } >"$WORK/README.tiny.md"
  write_readme tiny "$WORK/README.tiny.md"
}

# Q-410: the budget model. All 300 frames of `kitchen` in the n3dv form, at --resize 1.0 as
# the tiny runs and openftgs's own validation runs convert it (Q-T726.1); init and train on
# openftgs's defaults but for $BUDGET_CONFIG, seed 0; the export alone is kept.
make_budget() {
  : >"$COMMANDS"
  printf '%s\n' "$BUDGET_CONFIG" >"$WORK/kitchen_1m.config.json"
  local D=$WORK/kitchen_1m
  mkdir -p "$D"
  run "$FTGS" convert n3dv "$SYNTH/n3dv/kitchen" "$D/scene" --frames 300 --resize 1.0 \
    --jobs "$THREADS" >/dev/null
  run "$FTGS" init roma "$D/scene/scene.json" "$D/init.safetensors" --seed "$SEED"
  run "$FTGS" train "$D/scene/scene.json" --init "$D/init.safetensors" --out "$D/run" \
    --config "$WORK/kitchen_1m.config.json" --seed "$SEED"
  run "$FTGS" export "$D/run/checkpoint" --format ftgs-if --out "$OUT/kitchen_1m.safetensors"
  # The test split's scene-level metrics only: the per-image list has 300 entries.
  local METRICS
  METRICS=$("$PYTHON" -I -c '
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    print(json.dumps(m["entire"]["scene"], indent=1))
except (OSError, KeyError, ValueError):
    print("none")' "$D/run/METRICS.json")
  # The final primitive count and degree, from the export's header (ftgs-if, openftgs §2.2).
  local COUNT
  COUNT=$("$PYTHON" -I -c '
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
n, degree = h["position"]["shape"][0], h["__metadata__"]["sh_degree"]
print(f"{n:,} primitives at SH degree {degree}")' \
    "$OUT/kitchen_1m.safetensors")
  rm -rf "${D:?}"

  provenance
  {
    cat <<EOF
$BEGIN
## The budget model, \`kitchen_1m\` (Q-410)

Written by \`tools/fixtures/train_tiny.sh kitchen_1m\` on $(date -u +%Y-%m-%dT%H:%M:%SZ).
\`kitchen_1m.safetensors\` is openftgs-synth's \`kitchen\` trained at P-1's "1 M primitives at
degree 2" (§1.12), the model of B-1 with \`--model\` (P-2), B-2 and Dv-5. It has no request and
no reference renders; R-3 does not read it. #622's \`kitchen.safetensors\` is another model.

- Final count: $COUNT (\`growth.n_max\` 1,000,000).
- openftgs commit \`$COMMIT\` (\`$DESCRIBE\`${DIRTY:+, with local changes}), run as a program from its venv.
- Versions: $VERSIONS.
- openftgs-synth $SYNTH_VERSION at \`\$SYNTH\` (\`$SYNTH\`).
- GPU: $GPU (\`init\`, \`train\`). Every command ran with \`OMP_NUM_THREADS=$THREADS\`. Training
  on the GPU is not bitwise reproducible: a rerun gives another model.
- \`\$WORK\`, the intermediates (removed): the converted scene, frame cache, initial points and
  run directory.

Commands, in order:

\`\`\`sh
EOF
    cat "$COMMANDS"
    cat <<EOF
\`\`\`

\`\$WORK/kitchen_1m.config.json\`, its \`TrainConfig\` (TrainConfig's defaults otherwise):

\`\`\`json
$BUDGET_CONFIG
\`\`\`

Test metrics over the whole test split (\`entire.scene\` of \`ftgs train\`'s \`METRICS.json\`):

\`\`\`json
$METRICS
\`\`\`

SHA-256:

\`\`\`
EOF
    (cd "$OUT" && sha256sum kitchen_1m.safetensors)
    echo '```'
    echo "$END"
  } >"$WORK/README.kitchen_1m.md"
  write_readme kitchen_1m "$WORK/README.kitchen_1m.md"
}

((MAKE_TINY)) && make_tiny
((MAKE_BUDGET)) && make_budget
echo "wrote $OUT" >&2
