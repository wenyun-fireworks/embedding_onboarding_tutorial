"""Shared helpers for the PayFlow embedding onboarding demo.

The demo dataset is a tiny developer-docs retrieval task: user-style questions
(`queries.jsonl`) that must retrieve the reference passage that answers them
(`corpus.jsonl`), with one relevant passage per question (`qrels.tsv`). Query IDs
are partitioned into train / eval by `split.json`.
"""
from __future__ import annotations

import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def load_env(path: str | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (no python-dotenv needed).

    Existing environment variables win, so exported values are never overwritten.
    """
    path = path or os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)

# Qwen3-Embedding is instruction-aware: the query is prefixed with a task
# instruction, documents are encoded plain. This is the single most impactful
# knob for retrieval quality, so keep it identical between training and eval.
QUERY_INSTRUCTION = (
    "Given a developer question about the PayFlow payments API, "
    "retrieve the documentation passage that answers it"
)

# Must match the cookbook trainer's template (training/recipes/embedding_loop.py:
# "Instruct: {}\nQuery:" + query) so inference tokenization matches training.
_QUERY_TEMPLATE = "Instruct: {}\nQuery:"


def format_query(text: str) -> str:
    """Prefix a raw query with the task instruction, exactly as the trainer did."""
    return _QUERY_TEMPLATE.format(QUERY_INSTRUCTION) + text


def load_corpus(data_dir: str = DATA_DIR) -> dict[str, str]:
    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            title = r.get("title", "")
            corpus[r["_id"]] = f"{title}\n{r['text']}" if title else r["text"]
    return corpus


def load_queries(data_dir: str = DATA_DIR) -> dict[str, str]:
    with open(os.path.join(data_dir, "queries.jsonl")) as f:
        return {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}


def load_qrels(data_dir: str = DATA_DIR) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(os.path.join(data_dir, "qrels.tsv")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        for qid, did, score in r:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_split(data_dir: str = DATA_DIR) -> dict[str, list[str]]:
    with open(os.path.join(data_dir, "split.json")) as f:
        return json.load(f)
