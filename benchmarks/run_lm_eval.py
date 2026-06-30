"""MMLU / HellaSwag / ARC evaluation via lm-evaluation-harness.

Quantizes a model with Pare, then evaluates on standard LM tasks
using EleutherAI's lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness).

Install:
    pip install "lm-eval>=0.4.2,<0.4.4"   # 0.4.4+ requires Python 3.13

Usage:
    # Evaluate all schemes on Llama-2-7B
    python benchmarks/run_lm_eval.py --model llama2 --token $HF_TOKEN

    # Single scheme
    python benchmarks/run_lm_eval.py --model llama2 --method gptq-int4

Tasks evaluated:
    mmlu              57-subject multiple-choice QA (5-shot)
    hellaswag         sentence-completion commonsense (0-shot)
    arc_easy          ARC Easy multiple-choice (0-shot)
    arc_challenge     ARC Challenge multiple-choice (0-shot)

Results are appended to results/lm_eval_results.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from pare import QuantConfig, quantize

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "qwen":   "Qwen/Qwen2.5-7B",
}

METHODS: dict[str, QuantConfig | None] = {
    "fp16":        None,
    "rtn-int8":    QuantConfig(bits=8, scheme="rtn",         granularity="per_channel"),
    "gptq-int4":   QuantConfig(bits=4, scheme="gptq",        granularity="per_group", group_size=128),
    "awq-int4":    QuantConfig(bits=4, scheme="awq",         granularity="per_group", group_size=128),
    "smoothquant": QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel"),
}

TASKS = ["mmlu", "hellaswag", "arc_easy", "arc_challenge"]

DEVICE = "cuda"
SEQ_LEN = 2048
N_CALIB = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    parser.add_argument("--tasks",  default=",".join(TASKS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    args = parser.parse_args()

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[lm_eval] lm-evaluation-harness not installed. Run: pip install lm-eval")
        sys.exit(1)

    model_keys  = list(MODELS)  if args.model  == "all" else [args.model]
    method_keys = list(METHODS) if args.method == "all" else [args.method]
    tasks = [t.strip() for t in args.tasks.split(",")]

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "lm_eval_results.json"
    results: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    for model_key in model_keys:
        model_id = MODELS[model_key]
        print(f"\n[lm_eval] Loading {model_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=args.token, device_map="auto",
        )
        model_fp16.eval()

        # Calibration data (needed by GPTQ/AWQ/SmoothQuant)
        from datasets import load_dataset
        ds   = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
        toks = tokenizer(text, return_tensors="pt").input_ids
        calib_data = [toks[:, i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(N_CALIB)]

        for method_key in method_keys:
            tag = f"{model_key}/{method_key}"
            if tag in results:
                print(f"[lm_eval] {tag} already done — skipping")
                continue

            print(f"\n[lm_eval] {tag} ...", flush=True)
            config = METHODS[method_key]
            model_q = copy.deepcopy(model_fp16)
            model_q.eval()

            if config is not None:
                needs_calib = config.scheme in ("gptq", "awq", "smoothquant")
                quantize(
                    model_q, config,
                    calibration_data=calib_data if needs_calib else None,
                    device=DEVICE,
                )

            # Wrap for lm-evaluation-harness
            lm = HFLM(pretrained=model_q, tokenizer=tokenizer, batch_size=args.batch_size)
            eval_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot={"mmlu": 5, "hellaswag": 0, "arc_easy": 0, "arc_challenge": 0},
            )

            # Extract per-task accuracy
            task_scores = {}
            for task_name, task_result in eval_results["results"].items():
                # Primary metric differs by task: acc_norm for arc/hellaswag, acc for mmlu
                acc = task_result.get("acc_norm,none") or task_result.get("acc,none")
                task_scores[task_name] = round(acc * 100, 2) if acc else None

            results[tag] = task_scores
            print(f"[lm_eval] {tag}: {task_scores}", flush=True)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            del model_q
            torch.cuda.empty_cache()

        del model_fp16
        torch.cuda.empty_cache()

    print(f"\n[lm_eval] Done. Results at {results_path}")
    _print_summary(results)


def _print_summary(results: dict) -> None:
    print("\n" + "="*90)
    print(f"{'Model/Method':<30}  {'MMLU':>6}  {'HellaSwag':>10}  {'ARC-E':>6}  {'ARC-C':>6}")
    print("-"*90)
    for tag, scores in sorted(results.items()):
        if "error" in scores:
            print(f"{tag:<30}  ERROR")
            continue
        mmlu = f"{scores.get('mmlu', '—'):>6}"
        hella = f"{scores.get('hellaswag', '—'):>10}"
        arce  = f"{scores.get('arc_easy', '—'):>6}"
        arcc  = f"{scores.get('arc_challenge', '—'):>6}"
        print(f"{tag:<30}  {mmlu}  {hella}  {arce}  {arcc}")


if __name__ == "__main__":
    main()
