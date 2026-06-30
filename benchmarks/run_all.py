"""Full evaluation suite.

Runs all combinations of model × method, collecting PPL + throughput.
Results are written to results/eval_results.json.

Usage:
    cd /path/to/pare
    python benchmarks/run_all.py                   # all models, all methods
    python benchmarks/run_all.py --model llama2    # one model only
    python benchmarks/run_all.py --skip-throughput # PPL only (faster)

Models:
    llama2  = meta-llama/Llama-2-7b-hf   (correctness baseline)
    llama3  = meta-llama/Meta-Llama-3-8B  (modern arch, GQA)
    qwen    = Qwen/Qwen2.5-7B             (diversity; official GPTQ-Int4 for comparison)

Methods:
    fp16         baseline (no quantization)
    rtn-int8     RTN per-channel INT8
    gptq-int4    GPTQ per-group g=128 INT4
    awq-int4     AWQ per-group g=128 INT4
    smoothquant  SmoothQuant per-channel INT8 W+A

Results format (results/eval_results.json):
    {
      "<model>/<method>": {
        "wikitext2_ppl": float,
        "throughput": [{"batch_size": int, "tokens_per_sec": float, ...}, ...]
      }
    }
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from pare import QuantConfig, quantize
from pare.eval.perplexity import evaluate_perplexity
from pare.eval.throughput import benchmark_throughput

DEVICE    = "cuda"
SEQ_LEN   = 2048
N_CALIB   = 128
RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "qwen":   "Qwen/Qwen2.5-7B",
}

METHODS: dict[str, QuantConfig | None] = {
    "fp16":        None,   # no quantization
    "rtn-int8":    QuantConfig(bits=8, scheme="rtn",         granularity="per_channel"),
    "gptq-int4":   QuantConfig(bits=4, scheme="gptq",        granularity="per_group", group_size=128),
    "awq-int4":    QuantConfig(bits=4, scheme="awq",         granularity="per_group", group_size=128),
    "smoothquant": QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel"),
}


def load_model(model_id: str, token: str | None) -> tuple:
    print(f"\n[bench] Loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token, device_map="auto",
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[bench] Loaded  {vram:.1f} GB VRAM", flush=True)
    return model, tokenizer


def get_calibration_data(tokenizer, seq_len: int, n_calib: int) -> list:
    ds   = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    toks = tokenizer(text, return_tensors="pt").input_ids
    return [toks[:, i * seq_len : (i + 1) * seq_len] for i in range(n_calib)]


def run_one(
    model_key: str,
    method_key: str,
    model,
    tokenizer,
    calib_data: list,
    skip_throughput: bool,
) -> dict:
    config = METHODS[method_key]
    tag    = f"{model_key}/{method_key}"
    print(f"\n{'='*60}", flush=True)
    print(f"[bench] {tag}", flush=True)

    model_q = copy.deepcopy(model)
    model_q.eval()

    if config is not None:
        needs_calib = config.scheme in ("gptq", "awq", "smoothquant")
        t0 = time.time()
        quantize(
            model_q, config,
            calibration_data=calib_data if needs_calib else None,
            device=DEVICE,
        )
        print(f"[bench] Quantized in {time.time() - t0:.0f}s", flush=True)

    # PPL
    ppl = evaluate_perplexity(
        model_q, tokenizer, dataset="wikitext2",
        seq_len=SEQ_LEN, n_samples=None, device=DEVICE,
    )
    print(f"[bench] WikiText-2 PPL: {ppl:.2f}", flush=True)

    result: dict = {"wikitext2_ppl": ppl}

    # Throughput
    if not skip_throughput:
        print(f"[bench] Benchmarking throughput ...", flush=True)
        thr = benchmark_throughput(
            model_q, tokenizer,
            seq_len=256, n_generate=128,
            batch_sizes=[1, 4, 16, 32],
            device=DEVICE,
        )
        result["throughput"] = thr

    del model_q
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Pare evaluation suite")
    parser.add_argument("--model",  choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    args = parser.parse_args()

    model_keys  = list(MODELS)  if args.model  == "all" else [args.model]
    method_keys = list(METHODS) if args.method == "all" else [args.method]

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "eval_results.json"
    results: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    for model_key in model_keys:
        # Skip model download+load if every method for this model is already done.
        if all(f"{model_key}/{mk}" in results for mk in method_keys):
            print(f"[bench] {model_key} — all methods done, skipping", flush=True)
            continue

        model_id  = MODELS[model_key]
        model, tokenizer = load_model(model_id, args.token)
        calib_data = get_calibration_data(tokenizer, SEQ_LEN, N_CALIB)

        for method_key in method_keys:
            tag = f"{model_key}/{method_key}"
            if tag in results:
                print(f"[bench] {tag} already done — skipping", flush=True)
                continue

            try:
                r = run_one(model_key, method_key, model, tokenizer, calib_data, args.skip_throughput)
                results[tag] = r
            except Exception as e:
                print(f"[bench] {tag} FAILED: {e}", flush=True)
                results[tag] = {"error": str(e)}

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n[bench] All done. Results at {results_path}")
    _print_summary(results)


def _print_summary(results: dict) -> None:
    print("\n" + "="*70)
    print(f"{'Model/Method':<30}  {'WikiText-2 PPL':>15}  {'BS=1 tok/s':>12}")
    print("-"*70)
    for tag, r in sorted(results.items()):
        ppl = f"{r['wikitext2_ppl']:.2f}" if "wikitext2_ppl" in r else "ERR"
        thr = r.get("throughput", [{}])
        bs1 = next((t for t in thr if t.get("batch_size") == 1), {})
        tps = f"{bs1['tokens_per_sec']:.0f}" if bs1.get("tokens_per_sec") else "—"
        print(f"{tag:<30}  {ppl:>15}  {tps:>12}")


if __name__ == "__main__":
    main()
