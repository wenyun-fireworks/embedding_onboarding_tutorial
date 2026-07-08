#!/usr/bin/env python3
"""Turn the demo retrieval dataset into the trainer's (query, positive) JSONL.

The Fireworks embedding trainer takes one positive pair per line:
    {"query": "...", "positive": "..."}
In-batch negatives are generated automatically, so no negatives are needed.

We emit only the TRAIN split; the eval split is held out for measuring
before/after retrieval quality in step 6.
"""
from __future__ import annotations

import argparse
import json
import os

import common as C


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(C.DATA_DIR, "train_pairs.jsonl"))
    args = p.parse_args()

    queries = C.load_queries()
    corpus = C.load_corpus()
    qrels = C.load_qrels()
    train_ids = set(C.load_split()["train"])

    n = 0
    with open(args.out, "w") as f:
        for qid, docs in sorted(qrels.items()):
            if qid not in train_ids:
                continue
            for did in docs:
                f.write(json.dumps({"query": queries[qid], "positive": corpus[did]}) + "\n")
                n += 1
    print(f"wrote {n} (query, positive) pairs -> {args.out}")


if __name__ == "__main__":
    main()
