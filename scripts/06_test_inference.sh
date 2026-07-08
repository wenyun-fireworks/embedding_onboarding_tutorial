#!/usr/bin/env bash
# Step 6: test the deployed embedding model and measure the fine-tuning lift.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (see .env)}"
: "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID (see .env)}"
: "${EMBEDDING_MODEL_ID:?set EMBEDDING_MODEL_ID (see .env)}"
: "${DEPLOYMENT_ID:?set DEPLOYMENT_ID in .env (from step 5)}"
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/qwen3-embedding-8b}"
FT_REF="accounts/${FIREWORKS_ACCOUNT_ID}/models/${EMBEDDING_MODEL_ID}#accounts/${FIREWORKS_ACCOUNT_ID}/deployments/${DEPLOYMENT_ID}"

echo "=== raw /v1/embeddings smoke test ==="
curl -s -H "Authorization: Bearer $FIREWORKS_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${FT_REF}\",\"input\":[\"How do I refund a payment?\"]}" \
  "${FIREWORKS_BASE_URL:-https://api.fireworks.ai}/inference/v1/embeddings" | head -c 200
echo; echo

echo "=== retrieval on held-out queries: BASE vs FINE-TUNED ==="
"$PY" "$HERE/src/evaluate_endpoint.py" --model "$BASE_MODEL" --label base
"$PY" "$HERE/src/evaluate_endpoint.py" --model "$FT_REF" --label ft

echo "=== top-1 retrievals (fine-tuned) ==="
"$PY" "$HERE/src/test_inference.py" --model "$FT_REF" --n 8
