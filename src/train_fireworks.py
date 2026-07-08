#!/usr/bin/env python3
"""Fine-tune an embedding model on Fireworks via the Training SDK (no local GPU).

This is a thin wrapper around the public cookbook recipe
``training.recipes.embedding_loop``. The recipe provisions a trainer on
Fireworks infrastructure, streams the (query, positive) pairs, optimizes the
bidirectional in-batch-negative InfoNCE loss, saves a promotable checkpoint, and
(when --output-model-id is given) promotes that checkpoint to a model resource
you can download in the next step.

Prerequisites (see README):
  * FIREWORKS_API_KEY exported (or in .env).
  * The Fireworks cookbook on PYTHONPATH so `training.recipes.embedding_loop`
    imports, e.g.  export PYTHONPATH=/path/to/cookbook
  * `pip install tinker fireworks-ai`.
"""
from __future__ import annotations

import argparse
import os

import common as C

C.load_env()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-embedding-8b",
                   help="Fireworks-hosted embedding LLM to start from. Do NOT train from scratch.")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3-Embedding-8B",
                   help="HuggingFace tokenizer name matching the base model (client-side tokenization).")
    p.add_argument("--dataset", default=os.path.join(C.DATA_DIR, "train_pairs.jsonl"))
    p.add_argument("--output-model-id", default=os.environ.get("TRAINED_MODEL_ID", "qwen3-finetuned-trained"),
                   help="Promote the final checkpoint to this model id (downloaded + re-uploaded later).")
    p.add_argument("--output-mode", default="contrastive_loss",
                   choices=("embedding", "cos_similarity_matrix", "contrastive_loss"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=0, help="0 = full-parameter; 32 = cheap swappable adapter.")
    p.add_argument("--training-shape", default="", help="Blank lets the platform choose infra.")
    args = p.parse_args()

    # Imported here so `--help` works without the cookbook installed.
    import training.recipes.embedding_loop as embedding_loop
    from training.utils import TrainerConfig

    if "FIREWORKS_API_KEY" not in os.environ:
        raise SystemExit("FIREWORKS_API_KEY is not set (export it or put it in .env).")

    config = embedding_loop.Config(
        log_path="./embedding_logs",
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        dataset=args.dataset,
        output_mode=args.output_mode,
        # Task-specific query instruction (Qwen3 convention); documents get none.
        query_instruction=C.QUERY_INSTRUCTION,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape),
    )

    print(f"Starting Fireworks trainer: base={args.base_model} mode={args.output_mode} "
          f"epochs={args.epochs} batch={args.batch_size} -> model id '{args.output_model_id}'")
    metrics = embedding_loop.main(config)
    print("Training complete. Final metrics:", metrics)
    print(f"\nNext: download this model's checkpoint with\n"
          f"  firectl download model {args.output_model_id} ./export/{args.output_model_id}")


if __name__ == "__main__":
    main()
