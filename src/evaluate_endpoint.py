#!/usr/bin/env python3
"""Measure retrieval quality of a served embedding model on the held-out split.

Encodes the eval queries (with the task instruction) and the full corpus through
the Fireworks /v1/embeddings endpoint, ranks documents by cosine similarity, and
reports nDCG@10, Recall@10, and MRR with pytrec_eval. Run it once against the
base model and once against your fine-tuned deployment to see the lift.

    # base model (public, no deployment needed):
    python src/evaluate_endpoint.py --model accounts/fireworks/models/qwen3-embedding-8b --label base

    # fine-tuned deployment:
    python src/evaluate_endpoint.py \
        --model "accounts/<acct>/models/qwen3-finetuned#accounts/<acct>/deployments/<dep>" --label ft
"""
from __future__ import annotations

import argparse

import numpy as np
import pytrec_eval

import common as C
from fw_embed import embed

C.load_env()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Fireworks model/deployment resource name.")
    p.add_argument("--label", default="model")
    args = p.parse_args()

    queries = C.load_queries()
    corpus = C.load_corpus()
    qrels_all = C.load_qrels()
    eval_ids = set(C.load_split()["eval"])
    qrels = {q: d for q, d in qrels_all.items() if q in eval_ids}

    qids = sorted(qrels)
    dids = list(corpus)
    print(f"[{args.label}] eval queries={len(qids)}  corpus={len(dids)}  model={args.model}")

    q_emb = embed([C.format_query(queries[q]) for q in qids], args.model)
    d_emb = embed([corpus[d] for d in dids], args.model)

    sims = q_emb @ d_emb.T
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(qids):
        run[qid] = {dids[j]: float(sims[i, j]) for j in range(len(dids))}

    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10", "recip_rank"})
    per_q = ev.evaluate(run)
    keys = ("ndcg_cut_10", "recall_10", "recip_rank")
    means = {m: float(np.mean([v[m] for v in per_q.values()])) for m in keys}

    print(f"\n  [{args.label}]  nDCG@10={means['ndcg_cut_10']:.4f}  "
          f"Recall@10={means['recall_10']:.4f}  MRR={means['recip_rank']:.4f}\n")


if __name__ == "__main__":
    main()
