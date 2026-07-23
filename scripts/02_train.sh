#!/usr/bin/env bash
# Step 2: fine-tune on Fireworks via the Training SDK (no local GPU).
# Requires the Fireworks cookbook on PYTHONPATH so the embedding recipe imports.
#   export COOKBOOK_DIR=/path/to/cookbook   # https://github.com/fw-ai/cookbook
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
PY="${PY:-python3}"

if [ -z "${COOKBOOK_DIR:-}" ]; then
  echo "Set COOKBOOK_DIR to your local clone of https://github.com/fw-ai/cookbook" >&2
  exit 1
fi
export PYTHONPATH="${COOKBOOK_DIR}:${PYTHONPATH:-}"

# The recipe pools with pooling="last" and tokenizes with add_special_tokens=True,
# so training reads the LAST token's hidden state. To match the EMBEDDING serving
# path (which appends <|endoftext|> and last-token-pools), training MUST tokenize
# with the corresponding Qwen3-EMBEDDING tokenizer, whose post-processor appends
# <|endoftext|>. The base-LM tokenizer (e.g. Qwen/Qwen3-0.6B) does NOT append it,
# so pooling="last" would read the last CONTENT token instead — a train/serve
# mismatch. Derive the embedding-lineage tokenizer the same way steps 4 & 6 do
# (Qwen3-<size> -> Qwen3-Embedding-<size>); already-embedding names pass through.
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-Embedding-8B}"
case "$TOKENIZER_MODEL" in
  *Embedding*) TRAIN_TOKENIZER="$TOKENIZER_MODEL" ;;
  *)           TRAIN_TOKENIZER="$(printf '%s' "$TOKENIZER_MODEL" | sed 's#Qwen3-#Qwen3-Embedding-#')" ;;
esac

"$PY" "$HERE/src/train_fireworks.py" \
  --base-model "${BASE_MODEL:-accounts/fireworks/models/qwen3-embedding-8b}" \
  --tokenizer-model "$TRAIN_TOKENIZER" \
  --training-shape "${TRAINING_SHAPE:-}" \
  --lora-rank "${LORA_RANK:-0}" \
  --output-model-id "${TRAINED_MODEL_ID:-qwen3-finetuned-trained}" \
  --epochs "${EPOCHS:-15}" \
  --batch-size "${BATCH_SIZE:-8}"
