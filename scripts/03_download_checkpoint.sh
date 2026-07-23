#!/usr/bin/env bash
# Step 3: download the trained model's checkpoint to a local directory.
# The trainer promoted the final checkpoint to TRAINED_MODEL_ID; pull it so we
# can re-register it as an embedding model in the next step.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (see .env)}"
: "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID (see .env)}"
: "${TRAINED_MODEL_ID:?set TRAINED_MODEL_ID (see .env)}"

DEST="$HERE/export/${TRAINED_MODEL_ID}"
mkdir -p "$DEST"
firectl download model "$TRAINED_MODEL_ID" "$DEST" \
  --api-key "$FIREWORKS_API_KEY" -a "$FIREWORKS_ACCOUNT_ID"

echo
echo "Downloaded checkpoint -> $DEST"
ls -la "$DEST"
