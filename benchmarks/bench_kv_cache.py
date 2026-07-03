#!/usr/bin/env python3
"""
Benchmark: KIVI KV cache quantization vs FP16 baseline.

KIVI (Liu et al., ICML 2024) quantizes keys per-channel and values per-token,
exploiting the asymmetry that key channels have stable magnitudes while value
tokens do not.

Metrics:
  1. K and V reconstruction MSE at the SDPA call site
  2. Attention output relative L2 error
  3. WikiText-2 perplexity

Usage:
    python benchmarks/bench_kv_cache.py \\
        --model Qwen/Qwen2.5-7B --token $HF_TOKEN --seq-len 2048
"""

import argparse
import json
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pare.kv_cache import (
    _dequantize_keys, _dequantize_values,
    _quantize_keys, _quantize_values,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--token", default=None)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-ppl", type=int, default=64)
    p.add_argument("--n-mse", type=int, default=16)
    p.add_argument("--out", default="results/kv_cache_comparison.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# KIVI: per-channel keys, per-token values (Liu et al., ICML 2024)
# ---------------------------------------------------------------------------

def kivi_roundtrip(k, v, bits, group_size):
    T = k.shape[2]
    n = (T // group_size) * group_size
    if n == 0:
        return k, v
    kf, vf = k[:, :, :n].float(), v[:, :, :n].float()
    kq, ks, kz = _quantize_keys(kf, bits, group_size)
    kr = _dequantize_keys(kq, ks, kz, group_size).to(k.dtype)
    vq, vs, vz = _quantize_values(vf, bits, group_size)
    vr = _dequantize_values(vq, vs, vz).to(v.dtype)
    if n < T:
        kr = torch.cat([kr, k[:, :, n:]], dim=2)
        vr = torch.cat([vr, v[:, :, n:]], dim=2)
    return kr, vr


# ---------------------------------------------------------------------------
# SDPA patch — quantize K/V at the attention call site
# ---------------------------------------------------------------------------

@contextmanager
def patched_sdpa(bits=4, group_size=32, collect_stats=False):
    stats = {"k_mse": [], "v_mse": [], "attn_rel_err": []} if collect_stats else None
    _orig = F.scaled_dot_product_attention

    def _patched(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kwargs):
        key_q, val_q = kivi_roundtrip(key, value, bits, group_size)

        if collect_stats:
            with torch.no_grad():
                stats["k_mse"].append(((key.float() - key_q.float()) ** 2).mean().item())
                stats["v_mse"].append(((value.float() - val_q.float()) ** 2).mean().item())
                ref = _orig(query, key,   value,  attn_mask, 0.0, is_causal, scale=scale, **kwargs)
                out = _orig(query, key_q, val_q,  attn_mask, 0.0, is_causal, scale=scale, **kwargs)
                num = ((ref.float() - out.float()) ** 2).mean().item()
                den = (ref.float() ** 2).mean().item() + 1e-10
                stats["attn_rel_err"].append(num / den)
            return out

        return _orig(query, key_q, val_q, attn_mask, dropout_p, is_causal, scale=scale, **kwargs)

    F.scaled_dot_product_attention = _patched
    try:
        yield stats
    finally:
        F.scaled_dot_product_attention = _orig


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(model, token_ids, seq_len, n):
    nlls = []
    n_win = min(n, (token_ids.shape[1] - 1) // seq_len)
    for i in range(n_win):
        chunk = token_ids[:, i * seq_len:(i + 1) * seq_len].cuda()
        nlls.append(model(chunk, labels=chunk).loss.item())
    return math.exp(sum(nlls) / len(nlls))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  KV Cache Benchmark — KIVI")
    print(f"  {args.model}  |  {args.bits}-bit  |  group_size={args.group_size}")
    print(f"{sep}\n")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model…")
    tok = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        device_map="auto", token=args.token,
        attn_implementation="sdpa",
    )
    model.eval()

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  layers={cfg.num_hidden_layers}  "
          f"heads={cfg.num_attention_heads}  head_dim={head_dim}")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\nLoading WikiText-2 test…")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(raw["text"])
    token_ids = tok(text, return_tensors="pt").input_ids
    print(f"  {token_ids.shape[1]:,} tokens")

    # ── FP16 baseline PPL ─────────────────────────────────────────────────
    print(f"\n[PPL] FP16 baseline ({args.n_ppl} seqs × {args.seq_len} tokens)…")
    t0 = time.time()
    ppl_fp16 = perplexity(model, token_ids, args.seq_len, args.n_ppl)
    print(f"  FP16  PPL = {ppl_fp16:.4f}  ({time.time()-t0:.0f}s)")

    # ── KIVI ──────────────────────────────────────────────────────────────
    print(f"\n── KIVI ──")
    print(f"  [MSE] {args.n_mse} seqs × {args.seq_len} tokens…")
    t0 = time.time()
    with patched_sdpa(bits=args.bits, group_size=args.group_size,
                      collect_stats=True) as stats:
        with torch.no_grad():
            for i in range(args.n_mse):
                s = i * args.seq_len
                model(token_ids[:, s:s + args.seq_len].cuda())
    mse = {
        "k_mse":        sum(stats["k_mse"])        / len(stats["k_mse"]),
        "v_mse":        sum(stats["v_mse"])        / len(stats["v_mse"]),
        "attn_rel_err": sum(stats["attn_rel_err"]) / len(stats["attn_rel_err"]),
    }
    print(f"  K_MSE={mse['k_mse']:.3e}  V_MSE={mse['v_mse']:.3e}  "
          f"attn_rel_err={mse['attn_rel_err']:.3e}  ({time.time()-t0:.0f}s)")

    print(f"  [PPL] {args.n_ppl} seqs × {args.seq_len} tokens…")
    t0 = time.time()
    with patched_sdpa(bits=args.bits, group_size=args.group_size,
                      collect_stats=False):
        with torch.no_grad():
            ppl_kivi = perplexity(model, token_ids, args.seq_len, args.n_ppl)
    print(f"  PPL = {ppl_kivi:.4f}  (Δ {ppl_kivi - ppl_fp16:+.4f})  ({time.time()-t0:.0f}s)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {'Method':8}  {'K MSE':>9}  {'V MSE':>9}  "
          f"{'Attn err':>10}  {'PPL':>8}  {'ΔPPL':>8}")
    print(f"  {'-'*58}")
    print(f"  {'FP16':8}  {'—':>9}  {'—':>9}  {'—':>10}  "
          f"{ppl_fp16:>8.4f}  {'±0':>8}")
    print(f"  {'KIVI':8}  {mse['k_mse']:>9.3e}  {mse['v_mse']:>9.3e}  "
          f"{mse['attn_rel_err']:>10.3e}  {ppl_kivi:>8.4f}  {ppl_kivi - ppl_fp16:>+8.4f}")
    print(f"{sep}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "mse": {"kivi": mse},
            "ppl": {"fp16": ppl_fp16, "kivi": ppl_kivi},
        }, f, indent=2)
    print(f"Results → {args.out}")


if __name__ == "__main__":
    main()
