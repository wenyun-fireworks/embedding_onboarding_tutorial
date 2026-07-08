#!/usr/bin/env python3
"""Smoke-test a served embedding model: retrieve the top passage for a few queries.

Prints, for each held-out query, the top-ranked corpus passage and whether it is
the labeled-correct one. A quick human-readable confirmation that the deployment
serves embeddings and that retrieval is sensible.
"""
from __future__ import annotations

import argparse

import numpy as np

import common as C
from fw_embed import embed

C.load_env()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Fireworks model/deployment resource name.")
    p.add_argument("--n", type=int, default=5, help="How many held-out queries to show.")
    args = p.parse_args()

    queries = C.load_queries()
    corpus_raw = {r: t for r, t in C.load_corpus().items()}
    qrels = C.load_qrels()
    eval_ids = [q for q in C.load_split()["eval"] if q in qrels][: args.n]

    dids = list(corpus_raw)
    d_emb = embed([corpus_raw[d] for d in dids], args.model)
    q_emb = embed([C.format_query(queries[q]) for q in eval_ids], args.model)

    print(f"model={args.model}\n")
    hits = 0
    for i, qid in enumerate(eval_ids):
        sims = q_emb[i] @ d_emb.T
        top = dids[int(np.argmax(sims))]
        gold = set(qrels[qid])
        ok = top in gold
        hits += ok
        title = corpus_raw[top].split("\n", 1)[0]
        print(f"Q: {queries[qid]}")
        print(f"   top-1 -> {top} [{title}]  {'HIT' if ok else 'miss (gold=' + ','.join(gold) + ')'}")
    print(f"\ntop-1 accuracy on {len(eval_ids)} shown queries: {hits}/{len(eval_ids)}")


if __name__ == "__main__":
    main()
