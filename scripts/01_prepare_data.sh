#!/usr/bin/env bash
# Step 1: validate the demo dataset and build the trainer's (query, positive) JSONL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"

"$PY" "$HERE/src/prepare_data.py"
"$PY" "$HERE/src/make_train_pairs.py"
echo
echo "Train pairs ready at data/train_pairs.jsonl"
