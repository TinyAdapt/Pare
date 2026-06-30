"""Generate paper-ready LaTeX tables from pare benchmark results.

Reads:
    results/eval_results.json   — WikiText-2 PPL + throughput (from run_all.py)
    results/lm_eval_results.json  — MMLU / HellaSwag / ARC (from run_lm_eval.py)

Outputs two .tex files:
    results/table_accuracy.tex    — Table 1: PPL + downstream accuracy
    results/table_throughput.tex  — Table 2: throughput + VRAM

Usage:
    python benchmarks/make_tables.py
    python benchmarks/make_tables.py --stdout   # print to terminal instead
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Display order for methods
METHOD_LABELS = {
    "fp16":        "FP16 (baseline)",
    "rtn-int8":    "RTN INT8",
    "gptq-int4":   "GPTQ INT4 $g$=128",
    "awq-int4":    "AWQ INT4 $g$=128",
    "smoothquant": "SmoothQuant INT8 W+A",
}

MODEL_LABELS = {
    "llama2": "Llama-2-7B",
    "llama3": "Llama-3-8B",
    "qwen":   "Qwen2.5-7B",
}

METHODS = list(METHOD_LABELS)
MODELS  = list(MODEL_LABELS)


def load_results() -> tuple[dict, dict]:
    ppl_path = RESULTS_DIR / "eval_results.json"
    lme_path = RESULTS_DIR / "lm_eval_results.json"

    ppl_data = json.loads(ppl_path.read_text()) if ppl_path.exists() else {}
    lme_data = json.loads(lme_path.read_text()) if lme_path.exists() else {}
    return ppl_data, lme_data


def _fmt(val: float | None, decimals: int = 2, missing: str = "--") -> str:
    if val is None:
        return missing
    return f"{val:.{decimals}f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _best_quantized(col_vals: list[tuple[str, float | None]]) -> str | None:
    """Return the method key with the best (lowest for PPL, highest for acc) value,
    excluding fp16 baseline."""
    valid = [(k, v) for k, v in col_vals if k != "fp16" and v is not None]
    return None if not valid else min(valid, key=lambda x: x[1])[0]


def _best_quantized_high(col_vals: list[tuple[str, float | None]]) -> str | None:
    valid = [(k, v) for k, v in col_vals if k != "fp16" and v is not None]
    return None if not valid else max(valid, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Table 1: Accuracy (PPL + downstream)
# ---------------------------------------------------------------------------

def make_accuracy_table(ppl_data: dict, lme_data: dict) -> str:
    acc_tasks = ["mmlu", "hellaswag", "arc_easy", "arc_challenge"]
    acc_labels = ["MMLU", "HellaSwag", "ARC-E", "ARC-C"]

    n_acc = len(acc_tasks)
    col_spec = "ll" + "c" * (1 + n_acc)   # method + PPL + acc tasks

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{WikiText-2 perplexity ($\downarrow$) and downstream task accuracy ($\uparrow$, \%) "
        r"across three models and five quantization methods. "
        r"Best quantized result per column in \textbf{bold}.}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Model & Method & PPL$\downarrow$ & " + " & ".join(acc_labels) + r" \\",
        r"\midrule",
    ]

    for mi, model_key in enumerate(MODELS):
        if mi > 0:
            lines.append(r"\midrule")
        model_label = MODEL_LABELS[model_key]

        # Collect all values per column for bolding
        ppl_col  = []
        acc_cols = {t: [] for t in acc_tasks}
        for method_key in METHODS:
            tag = f"{model_key}/{method_key}"
            ppl_val  = ppl_data.get(tag, {}).get("wikitext2_ppl")
            ppl_col.append((method_key, ppl_val))
            for t in acc_tasks:
                acc_cols[t].append((method_key, lme_data.get(tag, {}).get(t)))

        best_ppl = _best_quantized(ppl_col)
        best_acc = {t: _best_quantized_high(acc_cols[t]) for t in acc_tasks}

        for ri, method_key in enumerate(METHODS):
            tag = f"{model_key}/{method_key}"
            method_label = METHOD_LABELS[method_key]
            row_model    = r"\multirow{" + str(len(METHODS)) + r"}{*}{" + model_label + "}" if ri == 0 else ""

            ppl_val = ppl_data.get(tag, {}).get("wikitext2_ppl")
            ppl_str = _fmt(ppl_val)
            if method_key == best_ppl:
                ppl_str = _bold(ppl_str)

            acc_strs = []
            for t in acc_tasks:
                v = lme_data.get(tag, {}).get(t)
                s = _fmt(v, decimals=1)
                if method_key == best_acc[t]:
                    s = _bold(s)
                acc_strs.append(s)

            sep = " & ".join([row_model, method_label, ppl_str] + acc_strs)
            lines.append(sep + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Throughput (tokens/sec + VRAM, relative speedup vs FP16)
# ---------------------------------------------------------------------------

def make_throughput_table(ppl_data: dict) -> str:
    batch_sizes = [1, 4, 16, 32]
    bs_labels   = [f"BS={b}" for b in batch_sizes]

    # Use first model as representative (Llama-2-7B)
    model_key = "llama2"

    col_spec = "l" + "cc" * len(batch_sizes)  # method + (tok/s, VRAM) per BS
    bs_header = " & ".join(
        r"\multicolumn{2}{c}{" + l + r"}" for l in bs_labels
    )
    sub_header = " & ".join(["Tok/s & VRAM (GB)"] * len(batch_sizes))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference throughput and peak GPU memory for Llama-2-7B "
        r"at varying batch sizes on an NVIDIA A40. "
        r"Speedup relative to FP16 in parentheses.}",
        r"\label{tab:throughput}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Method & " + bs_header + r" \\",
        r"\cmidrule(lr){2-3}" + "".join(
            r" \cmidrule(lr){" + str(2 + 2*i) + "-" + str(3 + 2*i) + "}"
            for i in range(1, len(batch_sizes))
        ),
        r"& " + sub_header + r" \\",
        r"\midrule",
    ]

    # Collect FP16 tok/s per batch size for speedup computation
    fp16_tps = {}
    fp16_tag = f"{model_key}/fp16"
    for entry in ppl_data.get(fp16_tag, {}).get("throughput", []):
        fp16_tps[entry["batch_size"]] = entry.get("tokens_per_sec")

    for method_key in METHODS:
        tag = f"{model_key}/{method_key}"
        thr = {e["batch_size"]: e for e in ppl_data.get(tag, {}).get("throughput", [])}

        cells = []
        for bs in batch_sizes:
            entry = thr.get(bs, {})
            tps  = entry.get("tokens_per_sec")
            vram = entry.get("peak_vram_gb")

            if tps is None:
                tps_str = "--"
            elif method_key != "fp16" and fp16_tps.get(bs):
                speedup = tps / fp16_tps[bs]
                tps_str = f"{tps:.0f} ({speedup:.2f}$\\times$)"
            else:
                tps_str = f"{tps:.0f}"

            vram_str = _fmt(vram, decimals=1) if vram else "--"
            cells += [tps_str, vram_str]

        lines.append(METHOD_LABELS[method_key] + " & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of files")
    args = parser.parse_args()

    ppl_data, lme_data = load_results()

    if not ppl_data and not lme_data:
        print("[make_tables] No results found. Run benchmarks/run_all.py and run_lm_eval.py first.")
        return

    acc_table = make_accuracy_table(ppl_data, lme_data)
    thr_table = make_throughput_table(ppl_data)

    if args.stdout:
        print("% === TABLE 1: ACCURACY ===\n")
        print(acc_table)
        print("\n\n% === TABLE 2: THROUGHPUT ===\n")
        print(thr_table)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "table_accuracy.tex").write_text(acc_table)
        (RESULTS_DIR / "table_throughput.tex").write_text(thr_table)
        print(f"[make_tables] Wrote results/table_accuracy.tex")
        print(f"[make_tables] Wrote results/table_throughput.tex")

    # Print a quick text summary of what data is available
    print("\n[make_tables] Available results:")
    all_tags = set(ppl_data) | set(lme_data)
    for model_key in MODELS:
        for method_key in METHODS:
            tag = f"{model_key}/{method_key}"
            has_ppl = tag in ppl_data and "wikitext2_ppl" in ppl_data[tag]
            has_thr = tag in ppl_data and "throughput" in ppl_data[tag]
            has_lme = tag in lme_data
            status  = []
            if has_ppl: status.append("PPL")
            if has_thr: status.append("throughput")
            if has_lme: status.append("lm_eval")
            if status:
                print(f"  {tag:<30} {', '.join(status)}")
            else:
                print(f"  {tag:<30} (missing)")


if __name__ == "__main__":
    main()
