#!/usr/bin/env python3
"""Verify input-form invariance of a deployed embedding model.

Contract we assert: embedding a **raw string** must return (numerically) the same
vector as embedding that string's **pre-computed token ids**. The OpenAI-style
``/v1/embeddings`` endpoint accepts ``input`` as a string, a list of strings, a
list of token ids, or a list of token-id lists; when you pass token ids the
server embeds them verbatim, so they must reproduce the server's own tokenization
of the raw text.

This holds for a model served as ``Kind: EMBEDDING_MODEL`` on the embedding
serving path (correct tokenization + pooling). A generative deployment
(``HF_BASE_MODEL``) does not guarantee it, so raw-text embeddings there can be
silently wrong — which is exactly what this check guards against.

Exits non-zero if any sample fails the cosine threshold, so step 6 hard-fails.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import requests

import common as C

C.load_env()


def _post_embeddings(inputs, model, api_key, base_url, timeout=120):
    """Call /v1/embeddings with an already-formed ``input`` payload (strings or
    token-id lists). Returns a list of raw (un-normalized) float vectors."""
    url = f"{base_url}/inference/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"model": model, "input": inputs}, timeout=timeout)
    if resp.status_code != 200:
        raise SystemExit(f"embeddings request failed ({resp.status_code}): {resp.text[:400]}")
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [np.asarray(d["embedding"], dtype=np.float64) for d in data]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="deployment ref: model#deployment")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer matching the model, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--threshold", type=float, default=0.9999,
                   help="min cosine(raw, ids) required to pass (default 0.9999)")
    p.add_argument("--n", type=int, default=4, help="number of corpus passages to test")
    args = p.parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("transformers is required for this check: pip install transformers")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Plain document passages (documents get no instruction prompt, so raw text
    # maps cleanly to its own token ids — no server-side prompt to replicate).
    corpus = C.load_corpus()
    texts = list(corpus.values())[: args.n] or ["How do I refund a payment?"]

    # Client tokenization must match the server's. A correctly-served Qwen3
    # embedding model applies the tokenizer's post-processor, which appends the
    # end-of-text token (<|endoftext|>, 151643) for last-token pooling — i.e.
    # add_special_tokens=True. We tokenize the same way and compare.
    raw_vecs = _post_embeddings(texts, args.model, api_key, base_url)

    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    print(f"model={args.model}")
    print(f"tokenizer={args.tokenizer} (<|endoftext|>={eot})")
    all_pass = True
    for i, (text, raw) in enumerate(zip(texts, raw_vecs)):
        ids = tok(text, add_special_tokens=True)["input_ids"]
        ids_vec = _post_embeddings([ids], args.model, api_key, base_url)[0]
        c = _cos(raw, ids_vec)
        rn = raw / (np.linalg.norm(raw) or 1.0)
        vn = ids_vec / (np.linalg.norm(ids_vec) or 1.0)
        max_abs = float(np.max(np.abs(rn - vn)))
        ok = c >= args.threshold
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        appends_eot = bool(ids and ids[-1] == eot)
        print(f"  [{status}] doc{i}: cos(raw, ids)={c:.6f} maxabs={max_abs:.2e} "
              f"(dim={raw.size}, ntok={len(ids)}, last_id={ids[-1]}, ends_with_eot={appends_eot}) "
              f"\"{text[:44]}...\"")

    print(f"\nINPUT-FORM INVARIANCE: {'PASS' if all_pass else 'FAIL'} "
          f"(threshold cos>={args.threshold})")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
