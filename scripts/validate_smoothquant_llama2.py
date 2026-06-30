"""Validate SmoothQuant INT8 W+A on Llama-2-7B-hf.


FP16 baseline for Llama-2-7B: 5.47 PPL.
Expected SmoothQuant INT8: +0.05–0.15 delta (better than GPTQ INT4 +0.26).

    
    python scripts/validate_smoothquant_llama2.py

Requires:
    - HF token set in HUGGING_FACE_HUB_TOKEN env var (model is gated)
    - transformers >= 4.46 (position_embeddings API)
    - ~14 GB VRAM for FP16 loading, ~8 GB after SmoothQuant INT8 quantization
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from datasets import load_dataset
from pare import QuantConfig, quantize
from pare.eval.perplexity import evaluate_perplexity

MODEL_ID   = "meta-llama/Llama-2-7b-hf"
DEVICE     = "cuda"
SEQ_LEN    = 2048
N_CALIB    = 128
N_EVAL_SEQ = None   # full WikiText-2 test set

def main():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGING_FACE_HUB_TOKEN env var to access gated model")

    print(f"[validate] Loading {MODEL_ID} in FP16 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, token=token, device_map="auto",
    )
    model.eval()
    vram_fp16 = torch.cuda.memory_allocated() / 1e9
    print(f"[validate] FP16 loaded: {vram_fp16:.1f} GB VRAM", flush=True)

    print("[validate] Preparing calibration data ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids
    calib_data = [tokens[:, i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(N_CALIB)]
    print(f"[validate] {len(calib_data)} calibration sequences × {SEQ_LEN} tokens", flush=True)

    config = QuantConfig(
        bits=8,
        scheme="smoothquant",
        granularity="per_channel",
        smooth_alpha=0.5,
    )

    print(f"\n[validate] Running SmoothQuant (α=0.5, INT8 per-channel) ...", flush=True)
    t0 = time.time()
    quantize(model, config, calibration_data=calib_data, device=DEVICE)
    elapsed = time.time() - t0
    vram_int8 = torch.cuda.memory_allocated() / 1e9
    print(f"[validate] SmoothQuant done in {elapsed:.0f}s  |  VRAM: {vram_int8:.1f} GB", flush=True)

    print("\n[validate] Evaluating PPL on WikiText-2 test ...", flush=True)
    ppl = evaluate_perplexity(
        model, tokenizer, dataset="wikitext2",
        seq_len=SEQ_LEN, n_samples=N_EVAL_SEQ, device=DEVICE,
    )

    print("\n" + "="*60)
    print(f"  Llama-2-7B  SmoothQuant INT8  WikiText-2 PPL: {ppl:.2f}")
    print(f"  FP16 baseline: 5.47  delta: +{ppl - 5.47:.2f}")
    print("="*60)

    if ppl <= 5.60:
        sys.exit(0)
    else:
        print("[validate] Gate missed — check smooth_alpha or calibration data.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
