"""CLI entry point: python -m pare <command> [options]

Commands
--------
quantize   Quantize a HuggingFace model and save to disk.
eval       Evaluate a quantized (or FP16) model's perplexity.

Examples
--------
# GPTQ INT4, 128 calibration sequences, save to ./llama2-7b-int4/
python -m pare quantize \\
    --model  meta-llama/Llama-2-7b-hf \\
    --bits   4 \\
    --scheme gptq \\
    --output ./llama2-7b-int4

# AWQ INT4 with custom alpha
python -m pare quantize \\
    --model  meta-llama/Llama-2-7b-hf \\
    --bits   4 \\
    --scheme awq \\
    --output ./llama2-7b-awq

# SmoothQuant INT8
python -m pare quantize \\
    --model  meta-llama/Llama-2-7b-hf \\
    --bits   8 \\
    --scheme smoothquant \\
    --output ./llama2-7b-sq-int8

# Evaluate a saved quantized model
python -m pare eval \\
    --model  meta-llama/Llama-2-7b-hf \\
    --quantized ./llama2-7b-int4 \\
    --dataset wikitext2

# Evaluate FP16 baseline (no --quantized flag)
python -m pare eval \\
    --model  meta-llama/Llama-2-7b-hf \\
    --dataset wikitext2
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pare",
        description="Pare — post-training quantization for LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── quantize ──────────────────────────────────────────────────────
    p_q = sub.add_parser("quantize", help="Quantize a model and save to disk")
    p_q.add_argument("--model",    required=True,  help="HuggingFace model ID or local path")
    p_q.add_argument("--output",   required=True,  help="Directory to save the quantized model")
    p_q.add_argument("--bits",     type=int, default=4, choices=[2, 3, 4, 8],
                     help="Target bit-width (default: 4)")
    p_q.add_argument("--scheme",   default="awq",
                     choices=["rtn", "gptq", "awq", "smoothquant"],
                     help="Quantization scheme (default: awq)")
    p_q.add_argument("--granularity", default="per_group",
                     choices=["per_tensor", "per_channel", "per_group"],
                     help="Scale granularity (default: per_group)")
    p_q.add_argument("--group-size", type=int, default=128,
                     help="Group size for per_group granularity (default: 128)")
    p_q.add_argument("--sym", action="store_true",
                     help="Symmetric quantization (no zero-point)")
    p_q.add_argument("--smooth-alpha", type=float, default=0.5,
                     help="SmoothQuant migration strength α (default: 0.5)")
    p_q.add_argument("--n-calib", type=int, default=128,
                     help="Number of calibration sequences (default: 128)")
    p_q.add_argument("--calib-seqlen", type=int, default=2048,
                     help="Length of each calibration sequence (default: 2048)")
    p_q.add_argument("--device", default="cuda",
                     help="Device for calibration forward passes (default: cuda)")
    p_q.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                     help="Model load dtype (default: float16)")
    p_q.add_argument("--token", default=None,
                     help="HuggingFace access token for gated models")

    # ── eval ──────────────────────────────────────────────────────────
    p_e = sub.add_parser("eval", help="Evaluate perplexity of a model")
    p_e.add_argument("--model",    required=True,  help="HuggingFace model ID or local path")
    p_e.add_argument("--quantized",   default=None,
                     help="Path to a Pare-saved quantized model (omit for FP16 baseline)")
    p_e.add_argument("--dataset",  default="wikitext2",
                     choices=["wikitext2", "c4"],
                     help="Evaluation dataset (default: wikitext2)")
    p_e.add_argument("--seq-len",  type=int, default=2048,
                     help="Sequence length (default: 2048)")
    p_e.add_argument("--n-samples", type=int, default=None,
                     help="Number of sequences to evaluate (default: full dataset)")
    p_e.add_argument("--device",   default="cuda")
    p_e.add_argument("--dtype",    default="float16", choices=["float16", "bfloat16", "float32"])
    p_e.add_argument("--token",    default=None)

    return parser


def cmd_quantize(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pare import QuantConfig, quantize, save_quantized

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[pare] Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, token=args.token, device_map="auto",
    )
    model.eval()

    config = QuantConfig(
        bits=args.bits,
        scheme=args.scheme,
        granularity=args.granularity,
        group_size=args.group_size,
        sym=args.sym,
        smooth_alpha=args.smooth_alpha,
    )

    calib_data = None
    if args.scheme != "rtn":
        print(f"[pare] Preparing {args.n_calib} calibration sequences ...", flush=True)
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer(text, return_tensors="pt").input_ids
        seq_len = args.calib_seqlen
        calib_data = [tokens[:, i * seq_len : (i + 1) * seq_len] for i in range(args.n_calib)]

    print(f"[pare] Quantizing with {args.scheme} INT{args.bits} ...", flush=True)
    quantize(model, config, calibration_data=calib_data, device=args.device)

    save_quantized(model, args.output)
    print(f"[pare] Done. Model saved to {args.output}", flush=True)


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pare import load_quantized
    from pare.eval.perplexity import evaluate_perplexity

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[pare] Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, token=args.token, device_map="auto",
    )
    model.eval()

    if args.quantized:
        print(f"[pare] Loading quantized weights from {args.quantized} ...", flush=True)
        load_quantized(model, args.quantized)

    print(f"[pare] Evaluating PPL on {args.dataset} ...", flush=True)
    ppl = evaluate_perplexity(
        model, tokenizer, dataset=args.dataset,
        seq_len=args.seq_len, n_samples=args.n_samples, device=args.device,
    )

    tag = f"{args.quantized or args.dtype}"
    print(f"\n  {args.model}  [{tag}]  {args.dataset} PPL: {ppl:.2f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
