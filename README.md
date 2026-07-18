# Full-Parameter Embedding Fine-Tuning on Fireworks — End-to-End

Minimal runbook: **prepare data → train → deploy → test**. Training runs on the
Fireworks Training SDK (GPU provisioned for you) via the cookbook recipe
`training.recipes.embedding_loop`. The trained checkpoint is deployed
**directly** — a dedicated deployment serves `/v1/embeddings` straight from it,
with no checkpoint download, no shard consolidation, and no re-upload as an
embedding model. Commands mirror the numbered scripts in `scripts/`.

## How it works

```
 data/ (query, positive) pairs
        │
        ▼
 ┌──────────────────┐   Fireworks Training SDK (embedding_loop recipe):
 │ 1. Prepare data  │   provisions a trainer, runs contrastive InfoNCE with
 │ 2. Train (SDK)   │   in-batch negatives, promotes the final checkpoint
 └──────────────────┘   to a model ($TRAINED_MODEL_ID)
        │  firectl create deployment  (deploy the trained model directly)
        ▼
 ┌──────────────────┐
 │ 3. Deploy        │   dedicated deployment serves /v1/embeddings straight from
 │                  │   the trained model — no download / reshard / re-upload
 └──────────────────┘
        │  /v1/embeddings
        ▼
 ┌──────────────────┐
 │ 4. Inference +   │   base vs fine-tuned retrieval metrics (nDCG@10, Recall@10, MRR)
 │    eval          │
 └──────────────────┘
```

## Why full-parameter embedding tuning

Fine-tuning the full model on your own **private / proprietary data** can yield **better embedding quality than an off-the-shelf open-source embedding model**. We have sufficient data to support this statement and we will share more details in the next a few weeks. 

That matters most when your domain's notion of "relevant" differs from general semantic similarity — e.g. which legal precedent governs a question, or which clinical trial a patient qualifies for.

- The flip side: on easy or broadly semantic tasks a strong base embedding model is already near the ceiling, so there is little left to gain (the demo task here is one such case — see [Notes](#notes)).

## Supported embedding models

This tutorial supports all three versions of the Qwen3 Embedding model family:

- Qwen3-Embedding-0.6B
- Qwen3-Embedding-4B
- Qwen3-Embedding-8B

## Prerequisites

- Fireworks account + API key; `firectl` on `PATH`.
- Python 3.10+. **Create and activate a virtualenv**, then install the deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

  `requirements.txt` installs **only** the local data-prep/eval tooling
  (numpy/requests/pytrec_eval/torch/etc.); `fireworks-ai`, `tinker`, and the
  cookbook are **not** installed by it and must be added separately (below) into
  the same venv that `PY` points at.

  The scripts run `$PY` (default `python3`), so if you used a virtualenv point
  `PY` at it (in `.env` or exported, e.g. `PY=$(pwd)/.venv/bin/python`) — see
  [Setup](#setup).
- `fireworks-ai`, `tinker`, and the [cookbook](https://github.com/fw-ai/cookbook)
  installed (the cookbook via `git clone` — see `requirements.txt`).
- A validated `POLICY_TRAINER` training shape for your base model (Step 0).

## Setup

The runnable scripts and all the code live in this repo, so you can dig into any
step: `scripts/` holds the numbered stages (`01_prepare_data.sh` …
`04_test_inference.sh`, run in order), `src/` holds the Python, and `.env.example`
(copy to `.env`) configures everything below.

Two local paths the scripts need — set them **in `.env`** or **export** them
before running the steps (`_load_env.sh` won't override an already-exported
value, so an exported value wins):

```bash
COOKBOOK_DIR=/path/to/cookbook          # your clone of https://github.com/fw-ai/cookbook (Step 2)
PY=/path/to/venv/bin/python             # the python where you installed the deps
```

`PY` defaults to `python3`, so if you installed the deps in a virtualenv set
`PY` (in `.env` or exported) — or activate the venv — otherwise the steps run
against the wrong interpreter and imports fail.

## Step 0 — Base model + validated training shape (public, pre-created)

All three Qwen3 Embedding bases are **public** (owned by `pyroworks`) and their
validated `POLICY_TRAINER` training shapes are **public** (owned by
`accounts/fireworks`) — so you can run this **in your own account** with no
`pyroworks` membership. Set `FIREWORKS_ACCOUNT_ID` to your own account and pick
one row below; the trained/embedding models you create land in your account.


| Model      | BASE_MODEL                                            | TRAINING_SHAPE                                                                  | TOKENIZER         |
| ---------- | ----------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------- |
| Qwen3-0.6B | `accounts/pyroworks/models/qwen3-embedding-base-0-6b` | `accounts/fireworks/trainingShapes/qwen3-embedding-base-0-6b/versions/yn5j1jk9` | `Qwen/Qwen3-0.6B` |
| Qwen3-4B   | `accounts/pyroworks/models/qwen3-embedding-base-4b`   | `accounts/fireworks/trainingShapes/qwen3-embedding-base-4b/versions/b5dzfhsp`   | `Qwen/Qwen3-4B`   |
| Qwen3-8B   | `accounts/pyroworks/models/qwen3-embedding-base-8b`   | `accounts/fireworks/trainingShapes/qwen3-embedding-base-8b/versions/e3oirzs4`   | `Qwen/Qwen3-8B`   |


These are **tunable bases** with a consistent tokenizer (the end-thinking token
`</think>` encodes as a single token), so the fine-tune inherits it and serves
directly with no tokenizer fix‑up. To stand up your own base from scratch you'd
register a tunable base and create + validate a matching `POLICY_TRAINER` shape.

## Data format

Training input is a small **BEIR-style retrieval set** in `data/` (a tiny demo
PayFlow API-docs corpus), from which Step 1 derives the trainer's
`(query, positive)` pairs:

- `corpus.jsonl` — one passage per line:
`{"_id": "d01", "title": "Creating a charge", "text": "Use the Charges endpoint …"}`
- `queries.jsonl` — one query per line:
`{"_id": "q01", "text": "How do I take a single card payment from a buyer?"}`
- `qrels.tsv` — TSV with header `query-id  corpus-id  score`, one label per line: `q01  d01  1`
- `split.json` — train/eval query-id lists: `{"train": ["q01", …], "eval": ["q04", …]}`

`scripts/01_prepare_data.sh` validates referential integrity and emits the actual
trainer input, `data/train_pairs.jsonl` — one positive pair per line, where the
positive is the passage `title` + `\n` + `text`:

```json
{"query": "How do I take a single card payment from a buyer?", "positive": "Creating a charge\nUse the Charges endpoint to collect a one-time payment from a customer. …"}
```

Only the **train** split is emitted; the eval split is held out for the
before/after retrieval metrics in Step 4. **In-batch negatives are generated
automatically** during training, so you never supply negatives. To use your own
data, replace the files in `data/` — or just drop in your own `train_pairs.jsonl`
with the same `{"query": ..., "positive": ...}` shape.

## Step 1 — Prepare data

```bash
bash scripts/01_prepare_data.sh   # → data/train_pairs.jsonl (22 train, 8 eval)
```

## Step 2 — Train

```bash
bash scripts/02_train.sh          # ~30 steps; promotes model $TRAINED_MODEL_ID (Kind HF_BASE_MODEL)
```

## Step 3 — Deploy the trained model directly

```bash
bash scripts/03_deploy.sh
```

Creates a **self-serve dedicated deployment** of the trained model with plain
`firectl create deployment` (no admin tooling). The trained checkpoint is
promoted as `Kind: HF_BASE_MODEL`, but a **dedicated deployment serves it on
`/v1/embeddings` directly** — the serving engine pools the model's hidden states,
so there is no need to download the checkpoint, consolidate shards, or re-upload
it as an embedding model. (Only **serverless** `/v1/embeddings` requires
`Kind: EMBEDDING_MODEL`; a dedicated deployment does not.)

Pick the deploy GPU via `ACCELERATOR_TYPE` in `.env` (default
`NVIDIA_H100_80GB`); if you hit capacity or quota limits, switch to another GPU.
`REGION` defaults to `us-iowa-1` (co-located with the GCS-backed model
artifacts) so the replica's model download doesn't stall on a far cluster; set it
to another region if needed, or empty for GLOBAL. Then grab the deployment id:

```bash
firectl list deployments -a "$FIREWORKS_ACCOUNT_ID"   # copy the id → DEPLOYMENT_ID in .env
```

Delete the deployment when done to stop billing (see [Cleanup](#cleanup)).

## Step 4 — Test

```bash
bash scripts/04_test_inference.sh   # /v1/embeddings smoke + base-vs-fine-tuned nDCG@10 / Recall@10 / MRR
```

The baseline is a strong off-the-shelf **serverless** embedding model
(`EVAL_BASE_MODEL`, default `accounts/fireworks/models/qwen3-embedding-8b`) —
**not** the tunable training `BASE_MODEL` from Step 2, which isn't served on
serverless `/v1/embeddings`. It's overridable via `EVAL_BASE_MODEL` and optional:
the base leg is non-fatal, so the fine-tuned metrics and top-1 retrievals still
print even if the baseline errors. This is the "beat off-the-shelf open-source"
comparison from the [Why](#why-full-parameter-embedding-tuning) section.

## Cleanup

```bash
# deployment first (billing); --ignore-checks if it has served requests
firectl delete deployment "$DEPLOYMENT_ID" -a "$FIREWORKS_ACCOUNT_ID" --ignore-checks
# then the model you created (the shared base can be kept for future fine-tunes)
firectl delete model "$TRAINED_MODEL_ID"    -a "$FIREWORKS_ACCOUNT_ID"
```

> Delete the deployment **first** and wait for it to reach `DELETED` before
> deleting the model — otherwise `firectl delete model` fails with
> `FailedPrecondition: cannot delete model with active deployments`.

## Notes

- The demo task is intentionally easy, so base and fine-tuned both score near 1.0 — it demonstrates the workflow. Large gains need a specialized-relevance corpus.

