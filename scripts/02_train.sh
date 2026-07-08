#!/usr/bin/env bash
# Step 2: fine-tune on Fireworks via the Training SDK (no local GPU).
# Requires the Fireworks cookbook on PYTHONPATH so the embedding recipe imports.
#   export COOKBOOK_DIR=/path/to/cookbook   # https://github.com/fw-ai/cookbook
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"

if [ -z "${COOKBOOK_DIR:-}" ]; then
  echo "Set COOKBOOK_DIR to your local clone of https://github.com/fw-ai/cookbook" >&2
  exit 1
fi
export PYTHONPATH="${COOKBOOK_DIR}:${PYTHONPATH:-}"

"$PY" "$HERE/src/train_fireworks.py" \
  --base-model "${BASE_MODEL:-accounts/fireworks/models/qwen3-embedding-8b}" \
  --tokenizer-model "${TOKENIZER_MODEL:-Qwen/Qwen3-Embedding-8B}" \
  --training-shape "${TRAINING_SHAPE:-}" \
  --lora-rank "${LORA_RANK:-0}" \
  --output-model-id "${TRAINED_MODEL_ID:-qwen3-finetuned-trained}" \
  --epochs "${EPOCHS:-15}" \
  --batch-size "${BATCH_SIZE:-8}"
