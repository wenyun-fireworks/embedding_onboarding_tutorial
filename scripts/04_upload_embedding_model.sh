#!/usr/bin/env bash
# Step 4: re-upload the downloaded checkpoint AS AN EMBEDDING MODEL.
# The --embedding flag sets the model kind to an embeddings base model
# (Kind: EMBEDDING_MODEL), which puts it on the correct, input-form-invariant
# embedding serving path. Without it the model would be treated as a generative
# model and raw-text embeddings can be subtly wrong (see README).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (see .env)}"
: "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID (see .env)}"
: "${TRAINED_MODEL_ID:?set TRAINED_MODEL_ID (see .env)}"
: "${EMBEDDING_MODEL_ID:?set EMBEDDING_MODEL_ID (see .env)}"
# Model metadata — derived from the chosen size so a 0.6B/4B model is not
# mislabeled as 8B. The fine-tuned model descends from the Qwen3-Embedding
# lineage, so point display name + HF url there (NOT the base-LM tokenizer repo,
# which is Qwen/Qwen3-<size>). All overridable via env.
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-Embedding-8B}"
case "$TOKENIZER_MODEL" in
  *Embedding*) EMB_LINEAGE="$TOKENIZER_MODEL" ;;                                  # already an embedding repo
  *)           EMB_LINEAGE="$(printf '%s' "$TOKENIZER_MODEL" | sed 's#Qwen3-#Qwen3-Embedding-#')" ;;
esac
DISPLAY_NAME="${DISPLAY_NAME:-${EMB_LINEAGE##*/} Fine-tuned Embedding}"
HF_URL="${HF_URL:-https://huggingface.co/${EMB_LINEAGE}}"

# firectl download nests the HF model files under
# export/<id>/tuned-model-<job>/<hash>/<id>/promoted-step-<n>-<hash>/hf/.
# Resolve the directory that actually contains config.json + safetensors.
CFG="$(find "$HERE/export/${TRAINED_MODEL_ID}" -name config.json -path '*/hf/*' 2>/dev/null | head -1)"
[ -z "$CFG" ] && CFG="$(find "$HERE/export/${TRAINED_MODEL_ID}" -name config.json 2>/dev/null | head -1)"
[ -z "$CFG" ] && { echo "no config.json under export/${TRAINED_MODEL_ID} — run 03_download_checkpoint.sh first" >&2; exit 1; }
UPLOAD_DIR="$(dirname "$CFG")"

# Drop the trainer-internal weight-spec file; deployable embedding models (public
# and our validated one) don't ship it. Keeps the uploaded file set clean.
rm -f "$UPLOAD_DIR/model.weight.spec.json"

# Note: no 1_Pooling/ directory is needed. The correct last-token pooling +
# <|endoftext|> tokenization is provided by the embedding DEPLOYMENT SHAPE at
# deploy time (step 5), not by files in the checkpoint (verified: uploading
# without 1_Pooling still passes the step-6 input-form invariance check).

echo "uploading from $UPLOAD_DIR"

firectl create model "$EMBEDDING_MODEL_ID" "$UPLOAD_DIR" \
  --embedding \
  --display-name "$DISPLAY_NAME" \
  --hugging-face-url "$HF_URL" \
  --api-key "$FIREWORKS_API_KEY" -a "$FIREWORKS_ACCOUNT_ID"

echo
echo "Created embedding model: accounts/${FIREWORKS_ACCOUNT_ID}/models/${EMBEDDING_MODEL_ID}"
echo "Wait for State: READY (firectl get model $EMBEDDING_MODEL_ID -a $FIREWORKS_ACCOUNT_ID)"
