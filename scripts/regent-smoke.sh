#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
upstream="$root/external/sts2ai"
assets="$root/artifacts/v1.0"
revision=d667489560cc46154638e1208ff032ed96707aa3
if [[ $(git -C "$upstream" rev-parse HEAD) != "$revision" ]]; then
    echo "Expected sts2ai v1.0 at $revision" >&2
    exit 1
fi
cd "$upstream"
seeds=CHECK00001,CHECK00002,CHECK00003,CHECK00004
train=(python3 scripts/train.py a10_longer --character regent --device cuda:0
    --init "$assets/sts2ai-weights-released-five/training_inputs/regent.pt"
    --seed 711091 --updates 2)
local_settings=(-- --n-envs 4 --steps-per-iter 1024 --fixed-update-rows 1024
    --minibatch 64 --token-budget 8192 --save-every 1)
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    "${train[@]}" --dry-run --out "$root/runs/preview/train" \
        --reserved-seeds "$root/runs/preview/reserved.json" "${local_settings[@]}"
    exit 0
fi
if (( $# != 0 )); then
    echo "Usage: bash scripts/regent-smoke.sh [--dry-run]" >&2
    exit 2
fi
for file in "$assets/regent.bin" "$assets/sts2ai-weights-released-five/training_inputs/regent.pt" \
    sim/bin/Release/net9.0/Sts2Sim.dll gamebin/v0.107.1/vocab.json; do
    [[ -s "$file" ]] || { echo "Missing: $file (see docs/regent.md)" >&2; exit 1; }
done
mkdir -p "$root/runs"
out=$(mktemp -d "$root/runs/regent-standard-A10-XXXXXXXX")
echo "Run directory: $out"
git rev-parse HEAD > "$out/upstream-commit.txt"
sha256sum "$assets/regent.bin" "$assets/sts2ai-weights-released-five/training_inputs/regent.pt" \
    gamebin/v0.107.1/{vocab.json,IMPORT_MANIFEST.json} > "$out/inputs.sha256"
PYTHONPATH=train python3 - "$seeds" > "$out/reserved.json" <<'PY'
import json
import sys
from sts2ai.seed_protocol import to_number

print(json.dumps([to_number(seed) for seed in sys.argv[1].split(",")]))
PY
eval_args=(--character regent --ascension 10 --workers 2 --seeds "$seeds")
python3 scripts/evaluate.py --model "$assets/regent.bin" "${eval_args[@]}" --out "$out/before"
{ time "${train[@]}" --out "$out/train" --reserved-seeds "$out/reserved.json" "${local_settings[@]}"; } \
    2>&1 | tee "$out/train.log"
python3 scripts/evaluate.py --model "$out/train/latest.pt" "${eval_args[@]}" --out "$out/after"
