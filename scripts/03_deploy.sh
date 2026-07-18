#!/usr/bin/env bash
# Step 3: deploy the TRAINED model directly (self-serve, non-admin path).
#
# The trained checkpoint from step 2 is promoted as Kind=HF_BASE_MODEL, but a
# dedicated deployment serves it on /v1/embeddings directly — the serving engine
# pools the model's hidden states, so NO re-registration as an embedding model
# and NO shard consolidation are needed. (Serverless /v1/embeddings is the only
# path that requires Kind=EMBEDDING_MODEL; a dedicated deployment does not.)
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

# Deploy GPU. Also NVIDIA_H200_141GB / NVIDIA_H100_80GB. Availability/quota
# varies per account: if the deploy stays CREATING with RESOURCE_EXHAUSTED or is
# rejected by a quota cap, set ACCELERATOR_TYPE to another GPU and retry.
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_H100_80GB}"
# Pin placement to a region co-located with your model's artifacts. Default
# GLOBAL placement can land the replica on a cluster where the cross-cloud model
# download stalls; a same-cloud region (e.g. us-iowa-1 for GCS artifacts) is more
# reliable. Leave REGION empty for GLOBAL.
REGION="${REGION:-us-iowa-1}"
REGION_FLAG=(); [ -n "$REGION" ] && REGION_FLAG=(--region "$REGION")

firectl create deployment "accounts/${FIREWORKS_ACCOUNT_ID}/models/${TRAINED_MODEL_ID}" \
  --accelerator-type "$ACCELERATOR_TYPE" --accelerator-count 1 \
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
