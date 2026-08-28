#!/usr/bin/env bash
# Step 3: deploy the fine-tuned model on the embedding serving path.
#
# IMPORTANT: deploy with an embedding DEPLOYMENT SHAPE (a validated preset owned
# by accounts/fireworks, named ...-minimal). The shape routes the model to the
# dedicated embedding serving path, which tokenizes with the model's EOS-appending
# post-processor (raw text -> "...<|endoftext|>") and last-token pooling — matching
# how the model was TRAINED (the recipe tokenizes with add_special_tokens=True).
# A plain deployment (no shape) runs the generative path and does NOT append
# <|endoftext|>, so raw-text embeddings would be wrong / not input-form invariant
# (step 4 checks this). The shape also selects the GPU/precision, so no
# --accelerator-type is needed.
#
# NOTE: a dedicated deployment is BILLABLE and the account must have a payment
# method configured, otherwise the create fails with
# "FailedPrecondition: payment method is required". Delete it when done (below).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/scripts/_load_env.sh"; _load_env "$HERE/.env"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (see .env)}"
: "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID (see .env)}"
: "${TRAINED_MODEL_ID:?set TRAINED_MODEL_ID (see .env)}"
: "${DEPLOYMENT_SHAPE:?set DEPLOYMENT_SHAPE (embedding shape, see README Step 0)}"

# Optional: pin placement to a region co-located with your model's artifacts. A
# same-cloud region (e.g. us-iowa-1 for GCS artifacts) can be more reliable than
# GLOBAL, but requires region-scoped GPU quota. Leave REGION empty for GLOBAL
# (needed when your account holds global rather than region-scoped quota). Only an
# *unset* REGION would fall back; an explicit empty REGION stays GLOBAL.
REGION="${REGION-}"
REGION_FLAG=(); [ -n "$REGION" ] && REGION_FLAG=(--region "$REGION")

firectl create deployment "accounts/${FIREWORKS_ACCOUNT_ID}/models/${TRAINED_MODEL_ID}" \
  --deployment-shape "$DEPLOYMENT_SHAPE" \
  --min-replica-count 1 --max-replica-count 1 \
  "${REGION_FLAG[@]}" \
  --wait \
  --api-key "$FIREWORKS_API_KEY" -a "$FIREWORKS_ACCOUNT_ID"

echo
echo "Deployment created. Find its id with:"
echo "  firectl list deployments -a $FIREWORKS_ACCOUNT_ID"
echo "Then set DEPLOYMENT_ID in .env before running 04_test_inference.sh."
echo "Remember to delete it when done to stop billing:"
echo "  firectl delete deployment <DEPLOYMENT_ID> -a $FIREWORKS_ACCOUNT_ID --ignore-checks"
