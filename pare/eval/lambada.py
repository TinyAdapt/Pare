"""LAMBADA accuracy evaluation.

LAMBADA tests whether a model can predict the last word of a passage when
given the full passage as context. It is more sensitive to quantization
collapse than PPL: an implementation with wrong scaling produces near-0%
accuracy, while a correct one degrades gracefully.

Metric: accuracy = fraction of examples where the model's greedy next-token
prediction matches the first token of the target word. This is a lower bound
on full-word accuracy but is deterministic and fast.

Reference: Paperno et al., "The LAMBADA dataset", 2016.
Dataset:   EleutherAI/lambada_openai (HuggingFace Hub)

Usage::

    from pare.eval.lambada import evaluate_lambada
    acc = evaluate_lambada(model, tokenizer, n_samples=200, device="cpu")
"""

from __future__ import annotations

import torch
import torch.nn as nn


def evaluate_lambada(
    model: nn.Module,
    tokenizer,
    n_samples: int | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """Compute LAMBADA accuracy (greedy next-token matching).

    Args:
        model:     Causal LM with HuggingFace-compatible forward signature.
        tokenizer: Matching tokenizer.
        n_samples: Number of examples to evaluate (None = full test set, ~5153).
                   Use 200-500 for fast smoke checks.
        device:    Device to run on.

    Returns:
        Accuracy in [0, 1].
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    model = model.to(device)
    model.eval()

    correct = 0
    for ex in ds:
        text: str = ex["text"]

        # Split at the last space — everything before is context.
        last_space = text.rfind(" ")
        if last_space == -1:
            continue
        context    = text[:last_space]
        target_str = text[last_space + 1:].strip()

        if not target_str:
            continue

        # Tokenize context; get next-token prediction.
        ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ctx_ids).logits  # [1, ctx_len, vocab]
        pred_id = int(logits[0, -1].argmax().item())

        # First token of the target (with a leading space, matching GPT-style vocab).
        target_ids = tokenizer(
            " " + target_str, add_special_tokens=False
        ).input_ids
        if not target_ids:
            continue
        target_first_id = target_ids[0]

        if pred_id == target_first_id:
            correct += 1

    return correct / len(ds)
