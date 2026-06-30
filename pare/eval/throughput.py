"""Throughput and memory benchmarking for quantized models.

Measures tokens/sec and peak GPU VRAM at multiple batch sizes.
Used to compare FP16 vs INT4/INT8 inference efficiency.

Usage::

    from pare.eval.throughput import benchmark_throughput
    results = benchmark_throughput(model, tokenizer, device="cuda")
    for r in results:
        print(f"bs={r['batch_size']}  {r['tokens_per_sec']:.0f} tok/s  {r['peak_vram_gb']:.1f} GB")
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch
import torch.nn as nn


def benchmark_throughput(
    model: nn.Module,
    tokenizer: Any,
    seq_len: int = 256,
    n_generate: int = 128,
    batch_sizes: list[int] | None = None,
    device: str | torch.device = "cuda",
    n_warmup: int = 3,
) -> list[dict]:
    """Measure autoregressive generation throughput and peak VRAM.

    Args:
        model:       Model to benchmark (FP16 or quantized).
        tokenizer:   Tokenizer for constructing input prompts.
        seq_len:     Number of prompt tokens fed as context.
        n_generate:  Number of new tokens to generate per run.
        batch_sizes: List of batch sizes to sweep (default: [1, 4, 16, 32]).
        device:      CUDA device to benchmark on.
        n_warmup:    Warmup runs before timing (excluded from measurement).

    Returns:
        List of dicts with keys: batch_size, tokens_per_sec, ms_per_token,
        peak_vram_gb, total_tokens.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32]

    device = torch.device(device)
    model.eval()

    # Build a fixed prompt of seq_len tokens.
    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10 + 1)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :seq_len]

    results = []
    for bs in batch_sizes:
        # Skip batch sizes that would OOM — catch and record None.
        try:
            result = _run_batch(model, input_ids, bs, n_generate, device, n_warmup)
            results.append(result)
            print(
                f"[pare] bs={bs:2d}  "
                f"{result['tokens_per_sec']:7.0f} tok/s  "
                f"{result['ms_per_token']:.1f} ms/tok  "
                f"{result['peak_vram_gb']:.2f} GB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[pare] bs={bs:2d}  OOM — skipped", flush=True)
            results.append({
                "batch_size": bs, "tokens_per_sec": None,
                "ms_per_token": None, "peak_vram_gb": None,
                "total_tokens": None, "oom": True,
            })
            torch.cuda.empty_cache()
            gc.collect()

    return results


def _run_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    batch_size: int,
    n_generate: int,
    device: torch.device,
    n_warmup: int,
) -> dict:
    """Time one (batch_size, n_generate) configuration."""
    batch_ids = input_ids.repeat(batch_size, 1).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(batch_ids, max_new_tokens=n_generate, do_sample=False)
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        model.generate(batch_ids, max_new_tokens=n_generate, do_sample=False)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    total_tokens    = batch_size * n_generate
    tokens_per_sec  = total_tokens / elapsed
    ms_per_token    = elapsed * 1000 / n_generate   # per-step latency
    peak_vram_gb    = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "batch_size":     batch_size,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token":   ms_per_token,
        "peak_vram_gb":   peak_vram_gb,
        "total_tokens":   total_tokens,
        "oom":            False,
    }


def format_results_table(
    fp16_results: list[dict],
    quantized_results: list[dict],
    label: str = "INT4",
) -> str:
    """Format a comparison table: FP16 vs quantized."""
    lines = [
        f"{'batch':>6}  {'FP16 tok/s':>11}  {'FP16 VRAM':>10}  "
        f"{label + ' tok/s':>11}  {label + ' VRAM':>10}  {'speedup':>8}",
        "-" * 68,
    ]
    for fp, qt in zip(fp16_results, quantized_results):
        bs = fp["batch_size"]
        if fp["tokens_per_sec"] is None or qt["tokens_per_sec"] is None:
            speedup_str = "OOM"
        else:
            speedup = qt["tokens_per_sec"] / fp["tokens_per_sec"]
            speedup_str = f"{speedup:.2f}×"

        fp_tok  = f"{fp['tokens_per_sec']:.0f}"  if fp["tokens_per_sec"] else "OOM"
        fp_mem  = f"{fp['peak_vram_gb']:.1f} GB" if fp["peak_vram_gb"]   else "OOM"
        qt_tok  = f"{qt['tokens_per_sec']:.0f}"  if qt["tokens_per_sec"] else "OOM"
        qt_mem  = f"{qt['peak_vram_gb']:.1f} GB" if qt["peak_vram_gb"]   else "OOM"

        lines.append(
            f"{bs:>6}  {fp_tok:>11}  {fp_mem:>10}  "
            f"{qt_tok:>11}  {qt_mem:>10}  {speedup_str:>8}"
        )
    return "\n".join(lines)
