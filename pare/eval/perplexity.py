"""Perplexity evaluation on WikiText-2 and C4.

Perplexity is the standard metric for LLM quantization papers:
    PPL = exp( -1/N * Σ log p(token_i | context) )

Lower is better. FP16 baseline is the reference; a good quantization
method should add < 0.5 PPL for INT8 and < 1.0 PPL for INT4 per-group.

Usage::

    from pare.eval.perplexity import evaluate_perplexity
    ppl = evaluate_perplexity(model, tokenizer, dataset="wikitext2")
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset: str = "wikitext2",
    seq_len: int = 2048,
    n_samples: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> float:
    """Compute perplexity on WikiText-2 or C4 test split.

    Args:
        model:     Any causal LM with a standard HuggingFace-compatible
                   forward signature (input_ids → logits).
        tokenizer: Matching tokenizer.
        dataset:   ``"wikitext2"`` or ``"c4"``.
        seq_len:   Context window length (stride = seq_len, no overlap).
        n_samples: Cap the number of sequences evaluated (None = all).
                   Useful for quick CPU smoke tests (e.g. n_samples=10).
        device:    Where to run evaluation.
        verbose:   Print per-batch NLL if True.

    Returns:
        Perplexity as a float.
    """
    model = model.to(device)
    model.eval()

    encodings = _load_and_encode(tokenizer, dataset)
    total_nll = 0.0
    total_tokens = 0
    seq_count = 0

    with torch.no_grad():
        n_seq = encodings.size(1) // seq_len
        if n_samples is not None:
            n_seq = min(n_seq, n_samples)

        for i in range(n_seq):
            chunk = encodings[:, i * seq_len : (i + 1) * seq_len].to(device)
            labels = chunk.clone()

            outputs = model(input_ids=chunk, labels=labels)
            nll = outputs.loss.item()   # mean NLL over this chunk

            total_nll += nll * seq_len
            total_tokens += seq_len
            seq_count += 1

            if verbose:
                print(f"  seq {i+1}/{n_seq}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}")

    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def _load_and_encode(tokenizer, dataset: str) -> Tensor:
    """Download and tokenize the test split, return as a flat token tensor."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets"
        )

    if dataset == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    elif dataset == "c4":
        data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        # Take first 1000 documents for a manageable eval
        text = "\n\n".join(d["text"] for d, _ in zip(data, range(1000)))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'wikitext2' or 'c4'.")

    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens
