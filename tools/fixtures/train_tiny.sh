#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The trained models of the `ds` lane (§2.7 Fixtures, D-29, §3.2 R-3): trains openftgs-synth's
# `tiny` and `tiny_unsync` scenes with openftgs, run as a program from its venv, and writes into
# $OPENFTGS_DATASETS/xr-fixtures/ (never committed), for NAME in tiny, tiny_unsync:
#
#   NAME.safetensors          the trained model's `ftgs export` (ftgs-if)
#   NAME.request.json         R-3's request, 20 items (tools/fixtures/r3_request.py)
#   NAME.renders.safetensors  `ftgs render ... --backend reference` of it, with the request's
#                             SHA-256
#   README.md                 the provenance: openftgs commit, versions, commands, config
#
# Everything else (the converted frames, the frame cache, the initial points, the run
# directory) goes to a work directory that is removed at the end: only the exports, requests and
# renders are kept. `kitchen` (B-2's budgets) is left to M-8.
#
# Usage: OPENFTGS_REPO=... OPENFTGS_DATASETS=... tools/fixtures/train_tiny.sh
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

[[ -x $FTGS ]] || { echo "$FTGS not found: install openftgs's venv" >&2; exit 2; }
[[ -f $SYNTH/DATASET_CARD.md ]] || { echo "$SYNTH is not openftgs-synth" >&2; exit 2; }

# Q-84: the training configuration, defaults but for the primitive cap (so the CPU
# reference renders stay short) and deterministic kernels.
CONFIG='{
  "growth": {
    "n_max": 100000
  },
  "perf": {
    "deterministic": true
  }
}'
SEED=0

mkdir -p "$OUT"
WORK=$(mktemp -d "$OUT/.work.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
COMMANDS=$WORK/commands.txt
: >"$COMMANDS"

# Run a command, recorded as the README shows it ($WORK and $OUT spelled as variables).
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

printf '%s\n' "$CONFIG" >"$WORK/config.json"

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

# The provenance.
VERSIONS=$("$PYTHON" -I -c '
import importlib.metadata as m, sys
names = ["openftgs", "ftgs-harness", "torch", "gsplat", "numpy", "safetensors", "pillow"]
print(", ".join(f"{n} {m.version(n)}" for n in names) + f"; Python {sys.version.split()[0]}")')
DIRTY=$(git -C "$OPENFTGS_REPO" status --porcelain)
{
  cat <<EOF
# openftgs-xr's trained fixtures (§2.7, D-29)

Written by openftgs-xr's \`tools/fixtures/train_tiny.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ); not
part of openftgs-synth and never committed. The \`ds\` lane's R-3 reads, for NAME in \`tiny\` and
\`tiny_unsync\`: \`NAME.safetensors\` (the trained model's \`ftgs export\`), \`NAME.request.json\` (R-3's
20 items, Q-83) and \`NAME.renders.safetensors\` (openftgs's reference renders, carrying the
request's SHA-256).

## Provenance

- openftgs commit \`$(git -C "$OPENFTGS_REPO" rev-parse HEAD)\` (\`$(git -C "$OPENFTGS_REPO" describe --tags --always)\`${DIRTY:+, with local changes}), run as a program from its venv.
- Versions: $VERSIONS.
- openftgs-synth $(cat "$SYNTH/VERSION" 2>/dev/null || echo unknown) at \`\$SYNTH\` (\`$SYNTH\`).
- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo none) (\`init\`, \`train\`); the
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
} >"$OUT/README.md"
echo "wrote $OUT" >&2
