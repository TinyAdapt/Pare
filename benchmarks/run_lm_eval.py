"""Zero-shot commonsense evaluation via lm-evaluation-harness.

Quantizes a model with Pare, then evaluates it on a set of zero-shot tasks
drawn from the SmoothQuant, GPTQ, and AWQ evaluation suites, namely LAMBADA,
PIQA, WinoGrande, OpenBookQA, RTE, and COPA.

Install:
    pip install "lm-eval>=0.4.12"
    # On Python <3.13, patch result_schema.py to remove extra_items= from TypedDict classes:
    #   sed -i 's/, extra_items=[^)]*)//' $(python -c "import lm_eval,os; print(os.path.join(os.path.dirname(lm_eval.__file__),'result_schema.py'))")

Usage:
    # Evaluate all schemes on Llama-3.1-8B
    python benchmarks/run_lm_eval.py --model llama31 --token $HF_TOKEN

    # Single scheme
    python benchmarks/run_lm_eval.py --model llama31 --method gptq-int4

Tasks evaluated:
    lambada_openai    next-word prediction (0-shot)
    piqa              physical commonsense reasoning (0-shot)
    winogrande        coreference resolution (0-shot)
    openbookqa        elementary science QA (0-shot)
    rte               textual entailment (0-shot)
    copa              causal reasoning (0-shot)

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
    "llama31": "meta-llama/Llama-3.1-8B",
    "qwen25":  "Qwen/Qwen2.5-7B",
    "olmo3":   "allenai/Olmo-3-1025-7B",
}

METHODS: dict[str, QuantConfig | None] = {
    "fp16":      None,
    "rtn-int8":  QuantConfig(bits=8, scheme="rtn",  granularity="per_channel"),
    "gptq-int4": QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=128),
    "awq-int4":  QuantConfig(bits=4, scheme="awq",  granularity="per_group", group_size=128),
}

# task -> num_fewshot. These six zero-shot tasks are from SmoothQuant Table 3.
TASK_FEWSHOT = {
    "lambada_openai": 0,
    "piqa":            0,
    "winogrande":      0,
    "openbookqa":      0,
    "rte":             0,
    "copa":            0,
}

DEVICE = "cuda"
SEQ_LEN = 2048
N_CALIB = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    parser.add_argument("--tasks",  default=",".join(TASK_FEWSHOT))
    parser.add_argument("--batch-size", default="auto")
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
        # Load on CPU so deepcopy doesn't double VRAM usage
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=args.token,
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
            # Deepcopy on CPU to avoid doubling VRAM; quantize() moves to GPU internally
            model_q = copy.deepcopy(model_fp16)
            model_q.eval()

            if config is not None:
                needs_calib = config.scheme in ("gptq", "awq", "smoothquant")
                quantize(
                    model_q, config,
                    calibration_data=calib_data if needs_calib else None,
                    device=DEVICE,
                )

            # Move to GPU for evaluation (model was deepcopied on CPU)
            model_q = model_q.to(DEVICE)

            # Wrap for lm-evaluation-harness
            # lm_eval 0.4.12+ requires num_fewshot as int; group requested tasks by
            # their fewshot count (TASK_FEWSHOT) and run one simple_evaluate() per group.
            lm = HFLM(pretrained=model_q, tokenizer=tokenizer, batch_size=args.batch_size)
            task_scores = {}
            fewshot_groups: dict[int, list[str]] = {}
            for t in tasks:
                fewshot_groups.setdefault(TASK_FEWSHOT.get(t, 0), []).append(t)

            for num_fewshot, group_tasks in fewshot_groups.items():
                r = lm_eval.simple_evaluate(model=lm, tasks=group_tasks, num_fewshot=num_fewshot)
                for task_name, task_result in r["results"].items():
                    acc = (
                        task_result.get("acc_norm,none")
                        or task_result.get("acc,none")
                        or task_result.get("exact_match,strict-match")
                        or task_result.get("exact_match,flexible-extract")
                    )
                    task_scores[task_name] = round(acc * 100, 2) if acc is not None else None

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


# The six zero-shot tasks from SmoothQuant Table 3, averaged the same way
# SmoothQuant reports its own accuracy number.
ZERO_SHOT_SUITE = ["lambada_openai", "piqa", "winogrande", "openbookqa", "rte", "copa"]


def _print_summary(results: dict) -> None:
    print("\n" + "="*55)
    print(f"{'Model/Method':<30}  {'ZeroShot Avg (6)':>16}")
    print("-"*55)
    for tag, scores in sorted(results.items()):
        if "error" in scores:
            print(f"{tag:<30}  ERROR")
            continue
        zs_scores = [scores[t] for t in ZERO_SHOT_SUITE if scores.get(t) is not None]
        zs_avg = f"{sum(zs_scores) / len(zs_scores):>16.2f}" if zs_scores else f"{'—':>16}"
        print(f"{tag:<30}  {zs_avg}")


if __name__ == "__main__":
    main()
