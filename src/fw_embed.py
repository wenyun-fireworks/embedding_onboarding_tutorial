"""Tiny client for the Fireworks /v1/embeddings endpoint (OpenAI-compatible)."""
from __future__ import annotations

import os

import numpy as np
import requests


def embed(texts: list[str], model: str, *, api_key: str | None = None,
          base_url: str | None = None, batch_size: int = 32, timeout: int = 120) -> np.ndarray:
    """Return L2-normalized embeddings for ``texts`` as a float32 [N, D] array.

    ``model`` is a Fireworks resource name, e.g.
    ``accounts/fireworks/models/qwen3-embedding-8b`` for the base model, or
    ``accounts/<acct>/models/<id>#accounts/<acct>/deployments/<dep>`` for a
    dedicated deployment.
    """
    api_key = api_key or os.environ["FIREWORKS_API_KEY"]
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    url = f"{base_url}/inference/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = requests.post(url, headers=headers, json={"model": model, "input": chunk}, timeout=timeout)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        vecs.extend(d["embedding"] for d in data)

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
