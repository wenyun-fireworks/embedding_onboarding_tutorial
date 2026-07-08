#!/usr/bin/env bash
# Step 4: re-upload the downloaded checkpoint AS AN EMBEDDING MODEL.
# The --embedding flag sets the model kind to an embeddings base model, which is
# what makes it servable on the /v1/embeddings endpoint. Without it, the model
# would be treated as a generative model and could not serve embeddings.
#
# Before uploading we CONSOLIDATE the checkpoint to a few large shards. A
# full-parameter fine-tune is saved as ~37 tiny safetensors shards; at deploy
# time the serving pod's Alluxio-backed download-models init container registers
# the whole file set in the coordinator's etcd in one request, and too many
# shards exceed etcd max-request-bytes (INVALID_ARGUMENT: request is too large),
# so the pod crash-loops and the deployment never goes healthy. ~4 large shards
# keep the file set small enough to mount cleanly.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (see .env)}"
: "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID (see .env)}"
: "${TRAINED_MODEL_ID:?set TRAINED_MODEL_ID (see .env)}"
: "${EMBEDDING_MODEL_ID:?set EMBEDDING_MODEL_ID (see .env)}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
# Model metadata — derived from the chosen base/tokenizer so a 0.6B/4B model is
# not mislabeled as 8B. Both are overridable via env.
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-Embedding-8B}"
DISPLAY_NAME="${DISPLAY_NAME:-${TOKENIZER_MODEL##*/} Fine-tuned Embedding}"
HF_URL="${HF_URL:-https://huggingface.co/${TOKENIZER_MODEL}}"
# Python with torch + safetensors + huggingface_hub for the reshard step.
RESHARD_PY="${RESHARD_PY:-${PY:-python3}}"

# firectl download nests the HF model files under
# export/<id>/tuned-model-<job>/<hash>/<id>/promoted-step-<n>-<hash>/hf/.
# Resolve the directory that actually contains config.json + safetensors.
CFG="$(find "$HERE/export/${TRAINED_MODEL_ID}" -name config.json -path '*/hf/*' 2>/dev/null | head -1)"
[ -z "$CFG" ] && CFG="$(find "$HERE/export/${TRAINED_MODEL_ID}" -name config.json 2>/dev/null | head -1)"
[ -z "$CFG" ] && { echo "no config.json under export/${TRAINED_MODEL_ID} — run 03_download_checkpoint.sh first" >&2; exit 1; }
SRC="$(dirname "$CFG")"

# Consolidate many small shards into ~4 large ones (skip if already few shards).
NSHARDS="$(ls -1 "$SRC"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')"
if [ "${NSHARDS:-0}" -le 6 ]; then
  echo "checkpoint already has ${NSHARDS} shard(s); skipping reshard"
  UPLOAD_DIR="$SRC"
else
  UPLOAD_DIR="$HERE/export/${EMBEDDING_MODEL_ID}-consolidated"
  echo "resharding ${NSHARDS} shards -> ~4 (max-shard-size ${MAX_SHARD_SIZE})"
  rm -rf "$UPLOAD_DIR"
  "$RESHARD_PY" "$HERE/src/reshard_checkpoint.py" \
    --src "$SRC" --out "$UPLOAD_DIR" --max-shard-size "$MAX_SHARD_SIZE"
fi

# Drop the trainer-internal weight-spec file; deployable embedding models (public
# and our validated one) don't ship it. Keeps the uploaded file set clean.
rm -f "$UPLOAD_DIR/model.weight.spec.json"

echo "uploading from $UPLOAD_DIR"

firectl create model "$EMBEDDING_MODEL_ID" "$UPLOAD_DIR" \
  --embedding \
  --display-name "$DISPLAY_NAME" \
  --hugging-face-url "$HF_URL" \
  --api-key "$FIREWORKS_API_KEY" -a "$FIREWORKS_ACCOUNT_ID"

echo
echo "Created embedding model: accounts/${FIREWORKS_ACCOUNT_ID}/models/${EMBEDDING_MODEL_ID}"
echo "Wait for State: READY (firectl get model $EMBEDDING_MODEL_ID -a $FIREWORKS_ACCOUNT_ID)"
