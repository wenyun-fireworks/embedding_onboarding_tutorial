#!/usr/bin/env python3
"""Validate the committed demo dataset and print its stats.

The demo data ships with the repo, so there is nothing to download. This script
just checks referential integrity (every qrel points at a real query/doc, the
train/eval split covers the labeled queries) and prints a summary so you can see
what the model is being trained and evaluated on.
"""
from __future__ import annotations

import sys

import common as C


def main() -> int:
    corpus = C.load_corpus()
    queries = C.load_queries()
    qrels = C.load_qrels()
    split = C.load_split()

    errors: list[str] = []
    for qid, docs in qrels.items():
        if qid not in queries:
            errors.append(f"qrel references unknown query {qid}")
        for did in docs:
            if did not in corpus:
                errors.append(f"qrel {qid} references unknown doc {did}")

    labeled = set(qrels)
    split_ids = set(split["train"]) | set(split["eval"])
    if labeled - split_ids:
        errors.append(f"labeled queries missing from split: {sorted(labeled - split_ids)}")
    overlap = set(split["train"]) & set(split["eval"])
    if overlap:
        errors.append(f"train/eval overlap: {sorted(overlap)}")

    print(f"corpus passages : {len(corpus)}")
    print(f"queries         : {len(queries)}")
    print(f"labeled queries : {len(labeled)}")
    print(f"train / eval     : {len(split['train'])} / {len(split['eval'])}")

    if errors:
        print("\nDATA VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nData looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
