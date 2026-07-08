#!/usr/bin/env python3
"""Consolidate a sharded HF safetensors checkpoint into a few large shards.

Why: a full-parameter fine-tune here is saved as ~37 tiny safetensors shards.
When such a model is deployed, the serving pod's Alluxio-backed `download-models`
init container registers the whole file set in the coordinator's etcd in a single
request. Too many shards blow past etcd `max-request-bytes`
(`INVALID_ARGUMENT: etcdserver: request is too large`), the init container
crash-loops, and the deployment never becomes healthy. Consolidating the weights
to ~4 large shards (5GB each) keeps the file set small enough to mount cleanly.

This script only rewrites the *weight* files. It:
  1. loads every `*.safetensors` shard in --src into one CPU state dict,
  2. writes consolidated shards + a fresh `model.safetensors.index.json` to --out
     via `huggingface_hub.save_torch_state_dict` (no model instantiation needed),
  3. copies every NON-weight file (config, tokenizer, etc.) from --src to --out.
"""
import argparse
import glob
import os
import shutil
import sys

from safetensors.torch import load_file
from huggingface_hub import save_torch_state_dict

# Weight files that this script regenerates; everything else in --src is aux
# metadata (config.json, tokenizer.*, modules.json, ...) and is copied verbatim.
_WEIGHT_INDEX = "model.safetensors.index.json"


def _shard_paths(src):
    return sorted(glob.glob(os.path.join(src, "*.safetensors")))


def _is_weight_file(name):
    # Old shard weights and the old index — regenerated, never copied.
    if name == _WEIGHT_INDEX:
        return True
    return name.endswith(".safetensors")


def reshard(src, out, max_shard_size):
    shards = _shard_paths(src)
    if not shards:
        sys.exit(f"no *.safetensors files found in {src}")
    os.makedirs(out, exist_ok=True)

    # Merge all shards into one state dict. Tensors stay on CPU; total footprint
    # is roughly the checkpoint size (~16 GB here), so this needs enough RAM.
    state_dict = {}
    for path in shards:
        for key, tensor in load_file(path).items():
            state_dict[key] = tensor
    print(f"loaded {len(state_dict)} tensors from {len(shards)} shard(s)")

    # Writes model-0000X-of-0000Y.safetensors + model.safetensors.index.json.
    save_torch_state_dict(state_dict, out, max_shard_size=max_shard_size)

    # Copy all non-weight files (config, tokenizer, sentence-transformers, etc.).
    copied = []
    for name in sorted(os.listdir(src)):
        full = os.path.join(src, name)
        if not os.path.isfile(full) or _is_weight_file(name):
            continue
        shutil.copy2(full, os.path.join(out, name))
        copied.append(name)
    print(f"copied {len(copied)} aux file(s): {', '.join(copied)}")

    new_shards = sorted(glob.glob(os.path.join(out, "*.safetensors")))
    print(f"wrote {len(new_shards)} consolidated shard(s) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source HF checkpoint dir")
    ap.add_argument("--out", required=True, help="output consolidated dir")
    ap.add_argument("--max-shard-size", default="5GB",
                    help="max size per output shard (e.g. 5GB); default 5GB")
    args = ap.parse_args()
    reshard(args.src, args.out, args.max_shard_size)


if __name__ == "__main__":
    main()
