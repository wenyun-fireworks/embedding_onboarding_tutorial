# Full-Parameter Embedding Fine-Tuning on Fireworks — End-to-End

Minimal runbook: **prepare data → train → deploy → test**. Training runs on the
Fireworks Training SDK (GPU provisioned for you) via the cookbook recipe
`training.recipes.embedding_loop`. Because the base is an `EMBEDDING_MODEL`, the
fine-tuned checkpoint is promoted as one too, so it deploys straight onto the
embedding serving path — which is what makes raw-text embeddings correct and
**input-form invariant** (see [Why the model kind matters](#why-the-model-kind-matters)).
Commands mirror the numbered scripts in `scripts/`.

## How it works

```
 data/ (query, positive) pairs
        │
        ▼
 ┌────────────────────────┐  Fireworks Training SDK (embedding_loop recipe):
 │ 1. Prepare data        │  provisions a trainer, runs contrastive InfoNCE with
 │ 2. Train (SDK)         │  in-batch negatives, promotes the final checkpoint
 └────────────────────────┘  to a model ($TRAINED_MODEL_ID, Kind EMBEDDING_MODEL)
        │  firectl create deployment
        ▼
 ┌────────────────────────┐
 │ 3. Deploy embedding    │  dedicated deployment w/ embedding deployment shape
 └────────────────────────┘  (...-minimal) -> embedding serving path, /v1/embeddings
        │  /v1/embeddings
        ▼
 ┌────────────────────────┐
 │ 4. Inference + eval    │  input-form invariance check (raw text == input_ids)
 │                        │  + base vs fine-tuned nDCG@10 / Recall@10 / MRR
 └────────────────────────┘
```

## Why the model kind matters

Embeddings served on the *generative* path are subtly wrong: raw-string input and
pre-tokenized `input_ids` are **not guaranteed to tokenize/pool identically**. The
correct, **input-form invariant** path is the dedicated embedding one, which
appends `<|endoftext|>` and applies last-token pooling — matching how the model was
trained (the recipe tokenizes with `add_special_tokens=True`).

Two things put your fine-tune on that path, and both are automatic here:

1. **The model kind.** A full-parameter fine-tune of a `Kind: EMBEDDING_MODEL`
   base is promoted as `EMBEDDING_MODEL` itself, so `$TRAINED_MODEL_ID` is
   servable as embeddings the moment training finishes. The Step 0 bases are all
   embedding-kind, so there is nothing to do. (This is also what serverless
   `/v1/embeddings` requires. Note it applies to full-parameter tuning; a LoRA run
   produces a PEFT addon instead.)
2. **The deployment shape.** A dedicated deployment only routes to the embedding
   serving path when created with an embedding **deployment shape** (Step 3). A
   plain deployment without one runs the generative path and skips the
   `<|endoftext|>` append, so raw-text embeddings come out wrong.

Step 4 asserts the resulting equivalence: embedding a raw string returns the same
vector as embedding that string's token ids.

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
- Python 3.11+. **Create and activate a virtualenv**, then install the deps:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install 'fireworks-ai[training]'
git clone https://github.com/fw-ai/cookbook
pip install -e cookbook/training
```

  `requirements.txt` installs **only** the local data-prep/eval tooling.
  Install the Training SDK through its `training` extra so it pins a compatible
  `tinker`; installing the latest `tinker` separately may be incompatible with
  the SDK. Install the cookbook's `training/` package (not the repository root)
  into the same venv that `PY` points at.

  The scripts run `$PY` (default `python3`), so if you used a virtualenv point
  `PY` at it (in `.env` or exported, e.g. `PY=$(pwd)/.venv/bin/python`) — see
  [Setup](#setup).
- `fireworks-ai[training]` and the
  [cookbook](https://github.com/fw-ai/cookbook) training package installed.
- A payment method on the account: Step 3 creates a billable dedicated deployment.

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

## Step 0 — Base model (public, pre-created)

All three Qwen3 Embedding bases are **public** (owned by `pyroworks`) and
registered as `Kind: EMBEDDING_MODEL`, so you can run this **in your own account**
with no `pyroworks` membership. Set `FIREWORKS_ACCOUNT_ID` to your own account and
pick one row below; the fine-tuned model you create lands in your account and
inherits the embedding kind.


| Model      | BASE_MODEL                                              | TOKENIZER_MODEL                  |
| ---------- | ------------------------------------------------------- | -------------------------------- |
| Qwen3-0.6B | `accounts/pyroworks/models/qwen3-embedding-0-6b-ft-base` | `Qwen/Qwen3-Embedding-0.6B`      |
| Qwen3-4B   | `accounts/pyroworks/models/qwen3-embedding-4b-ft-base`   | `Qwen/Qwen3-Embedding-4B`        |
| Qwen3-8B   | `accounts/pyroworks/models/qwen3-embedding-8b-ft-base`   | `Qwen/Qwen3-Embedding-8B`        |

Leave `TRAINING_SHAPE` empty (the default). The Training SDK then selects a
current validated shape compatible with `BASE_MODEL`. Pin a shape only when its
snapshot `base_model` exactly matches `BASE_MODEL`; stale shapes can fail during
trainer startup in the `regional-model-artifacts` container.


These are **tunable embedding bases** whose vocab matches the Qwen3-Embedding
tokenizer, so the fine-tune serves directly with no tokenizer fix‑up. Use the
**`Qwen/Qwen3-Embedding-<size>`**
tokenizer (above), **not** the base-LM `Qwen/Qwen3-<size>`: only the embedding
tokenizer's post-processor appends `<|endoftext|>` with `add_special_tokens=True`,
so the recipe's `pooling="last"` trains on the EOS token and matches the embedding
serving path. To stand up your own base from scratch you'd register it with
`firectl create model --embedding` and create + validate a matching
`POLICY_TRAINER` shape.

Step 3 deploys with a public **embedding deployment shape** (also owned by
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
bash scripts/02_train.sh          # ~30 steps; promotes model $TRAINED_MODEL_ID (Kind EMBEDDING_MODEL)
```

The promoted model inherits `Kind: EMBEDDING_MODEL` from the base, so it is ready
to deploy as-is (see [Why…](#why-the-model-kind-matters)). Confirm before moving
on:

```bash
firectl get model "$TRAINED_MODEL_ID" -a "$FIREWORKS_ACCOUNT_ID"   # State: READY, Kind: EMBEDDING_MODEL
```

## Step 3 — Deploy the fine-tuned model

```bash
bash scripts/03_deploy.sh
```

Creates a **self-serve dedicated deployment** of `$TRAINED_MODEL_ID` using an
**embedding deployment shape** (`DEPLOYMENT_SHAPE` in `.env`, the `...-minimal`
preset for your size). The shape is what routes the model to the embedding
serving path that appends `<|endoftext|>` and applies last-token pooling — i.e.
it makes raw-text embeddings **input-form invariant** and consistent with how the
model was trained (Step 4 verifies this). A plain deployment without a shape runs
the generative path and does **not** append `<|endoftext|>`, producing wrong
embeddings. The shape also selects the GPU/precision (no `ACCELERATOR_TYPE`
needed); leave `REGION` empty for GLOBAL. Then grab the deployment id:

```bash
firectl list deployments -a "$FIREWORKS_ACCOUNT_ID"   # copy the id → DEPLOYMENT_ID in .env
```

Delete the deployment when done to stop billing (see [Cleanup](#cleanup)).

## Step 4 — Test

```bash
bash scripts/04_test_inference.sh   # invariance check + base-vs-fine-tuned nDCG@10 / Recall@10 / MRR
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
# then the model you created (the shared base can be kept for future fine-tunes)
firectl delete model "$TRAINED_MODEL_ID"    -a "$FIREWORKS_ACCOUNT_ID"
```

> Delete the deployment **first** and wait for it to reach `DELETED` before
> deleting the model — otherwise `firectl delete model` fails with
> `FailedPrecondition: cannot delete model with active deployments`.

## Notes

- The demo task is intentionally easy, so base and fine-tuned both score near 1.0 — it demonstrates the workflow. Large gains need a specialized-relevance corpus.

