# Full-Parameter Embedding Fine-Tuning on Fireworks — End-to-End

Minimal runbook: **prepare data → train → download → upload as embedding model →
deploy → test**. Training runs on the Fireworks Training SDK (GPU provisioned for
you) via the cookbook recipe `training.recipes.embedding_loop`. The trained
checkpoint is then **re-registered as an `EMBEDDING_MODEL`** and deployed on the
embedding serving path — this is what makes raw-text embeddings correct and
**input-form invariant** (see [Why re-register…](#why-re-register-as-an-embedding_model)).
Commands mirror the numbered scripts in `scripts/`.

## How it works

```
 data/ (query, positive) pairs
        │
        ▼
 ┌────────────────────────┐  Fireworks Training SDK (embedding_loop recipe):
 │ 1. Prepare data        │  provisions a trainer, runs contrastive InfoNCE with
 │ 2. Train (SDK)         │  in-batch negatives, promotes the final checkpoint
 └────────────────────────┘  to a model ($TRAINED_MODEL_ID, Kind HF_BASE_MODEL)
        │  firectl download model
        ▼
 ┌────────────────────────┐
 │ 3. Download checkpoint  │  pull the trained HF checkpoint locally
 │ 4. Upload as EMBEDDING  │  `firectl create model --embedding`
 └────────────────────────┘  -> $EMBEDDING_MODEL_ID (Kind EMBEDDING_MODEL)
        │  firectl create deployment
        ▼
 ┌────────────────────────┐
 │ 5. Deploy embedding    │  dedicated deployment w/ embedding deployment shape
 └────────────────────────┘  (...-minimal) -> embedding serving path, /v1/embeddings
        │  /v1/embeddings
        ▼
 ┌────────────────────────┐
 │ 6. Inference + eval    │  input-form invariance check (raw text == input_ids)
 │                        │  + base vs fine-tuned nDCG@10 / Recall@10 / MRR
 └────────────────────────┘
```

## Why re-register as an `EMBEDDING_MODEL`

The trainer promotes the fine-tuned checkpoint as `Kind: HF_BASE_MODEL` (a
*generative* base). You can deploy that directly and it will answer
`/v1/embeddings`, **but** its embeddings are produced on the generative serving
path, where raw-string input and pre-tokenized `input_ids` are **not guaranteed
to tokenize/pool identically** — so raw-text embeddings can be subtly wrong.

Re-registering the same weights with `--embedding` (`Kind: EMBEDDING_MODEL`) and
deploying with an **embedding deployment shape** (Step 5) puts them on the
dedicated **embedding serving path**, which appends `<|endoftext|>` and applies
last-token pooling — matching how the model was trained (the recipe tokenizes with
`add_special_tokens=True`). This makes it **input-form invariant**: embedding a
raw string returns the same vector as embedding that string's token ids. Step 6
asserts this equivalence. (`Kind: EMBEDDING_MODEL` is also what serverless
`/v1/embeddings` requires.)

Both parts matter: `--embedding` sets the kind, but the **deployment shape** is
what routes a dedicated deployment to the embedding serving path. A plain
deployment (no shape) runs the generative path and skips the `<|endoftext|>`
append, so raw-text embeddings come out wrong.

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
`06_test_inference.sh`, run in order), `src/` holds the Python, and `.env.example`
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


| Model      | BASE_MODEL                                              | TRAINING_SHAPE                                                                  | TOKENIZER         |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------- |
| Qwen3-0.6B | `accounts/pyroworks/models/qwen3-embedding-0-6b-ft-base` | `accounts/fireworks/trainingShapes/qwen3-embedding-base-0-6b/versions/yn5j1jk9` | `Qwen/Qwen3-Embedding-0.6B` |
| Qwen3-4B   | `accounts/pyroworks/models/qwen3-embedding-4b-ft-base`   | `accounts/fireworks/trainingShapes/qwen3-embedding-base-4b/versions/b5dzfhsp`   | `Qwen/Qwen3-Embedding-4B`   |
| Qwen3-8B   | `accounts/pyroworks/models/qwen3-embedding-8b-ft-base`   | `accounts/fireworks/trainingShapes/qwen3-embedding-base-8b/versions/e3oirzs4`   | `Qwen/Qwen3-Embedding-8B`   |


These are **tunable bases** whose vocab matches the Qwen3-Embedding tokenizer, so
the fine-tune serves directly with no tokenizer fix‑up. Use the **`Qwen/Qwen3-Embedding-<size>`**
tokenizer (above), **not** the base-LM `Qwen/Qwen3-<size>`: only the embedding
tokenizer's post-processor appends `<|endoftext|>` with `add_special_tokens=True`,
so the recipe's `pooling="last"` trains on the EOS token and matches the embedding
serving path. To stand up your own base from scratch you'd register a tunable base
and create + validate a matching `POLICY_TRAINER` shape.

Step 5 deploys with a public **embedding deployment shape** (also owned by
`accounts/fireworks`); pick the one matching your size:

| Model      | DEPLOYMENT_SHAPE                                                     |
| ---------- | ------------------------------------------------------------------- |
| Qwen3-0.6B | `accounts/fireworks/deploymentShapes/qwen3-embedding-0p6b-minimal`  |
| Qwen3-4B   | `accounts/fireworks/deploymentShapes/qwen3-embedding-4b-minimal`    |
| Qwen3-8B   | `accounts/fireworks/deploymentShapes/qwen3-embedding-8b-minimal`    |

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
before/after retrieval metrics in Step 6. **In-batch negatives are generated
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

## Step 3 — Download the trained checkpoint

```bash
bash scripts/03_download_checkpoint.sh   # → export/$TRAINED_MODEL_ID/...
```

Pulls the trained checkpoint (promoted as `Kind: HF_BASE_MODEL`) to a local
`export/` dir so it can be re-registered as an embedding model next.

> Requires model-download access on your account. If `firectl download model`
> returns `FailedPrecondition: model downloading is restricted`, request access
> from Fireworks.

## Step 4 — Upload as an embedding model

```bash
bash scripts/04_upload_embedding_model.sh   # firectl create model --embedding
```

Re-uploads the downloaded checkpoint with `--embedding`, creating
`$EMBEDDING_MODEL_ID` (`Kind: EMBEDDING_MODEL`) — the kind that serves on the
correct, input-form invariant embedding path (see
[Why…](#why-re-register-as-an-embedding_model)). Wait for `State: READY`:

```bash
firectl get model "$EMBEDDING_MODEL_ID" -a "$FIREWORKS_ACCOUNT_ID"
```

## Step 5 — Deploy the embedding model

```bash
bash scripts/05_deploy.sh
```

Creates a **self-serve dedicated deployment** of `$EMBEDDING_MODEL_ID` using an
**embedding deployment shape** (`DEPLOYMENT_SHAPE` in `.env`, the `...-minimal`
preset for your size). The shape is what routes the model to the embedding
serving path that appends `<|endoftext|>` and applies last-token pooling — i.e.
it makes raw-text embeddings **input-form invariant** and consistent with how the
model was trained (Step 6 verifies this). A plain deployment without a shape runs
the generative path and does **not** append `<|endoftext|>`, producing wrong
embeddings. The shape also selects the GPU/precision (no `ACCELERATOR_TYPE`
needed); leave `REGION` empty for GLOBAL. Then grab the deployment id:

```bash
firectl list deployments -a "$FIREWORKS_ACCOUNT_ID"   # copy the id → DEPLOYMENT_ID in .env
```

Delete the deployment when done to stop billing (see [Cleanup](#cleanup)).

## Step 6 — Test

```bash
bash scripts/06_test_inference.sh   # invariance check + base-vs-fine-tuned nDCG@10 / Recall@10 / MRR
```

Runs three things:
1. a raw `/v1/embeddings` smoke test;
2. an **input-form invariance** check (`src/check_input_invariance.py`) — asserts
   that embedding a raw string returns the same vector as embedding that string's
   tokenized `input_ids` (hard-fails on mismatch);
3. base-vs-fine-tuned retrieval metrics.

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
firectl delete deployment "$DEPLOYMENT_ID"  -a "$FIREWORKS_ACCOUNT_ID" --ignore-checks
# then the models you created (the shared base can be kept for future fine-tunes)
firectl delete model "$EMBEDDING_MODEL_ID"  -a "$FIREWORKS_ACCOUNT_ID"
firectl delete model "$TRAINED_MODEL_ID"    -a "$FIREWORKS_ACCOUNT_ID"
```

> Delete the deployment **first** and wait for it to reach `DELETED` before
> deleting the models — otherwise `firectl delete model` fails with
> `FailedPrecondition: cannot delete model with active deployments`.

## Notes

- The demo task is intentionally easy, so base and fine-tuned both score near 1.0 — it demonstrates the workflow. Large gains need a specialized-relevance corpus.

