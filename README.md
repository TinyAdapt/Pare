# Pare

Quantize any LLM in one line. Switch between GPTQ, AWQ, SmoothQuant, and RTN by changing a config field.

---

## Benchmarks

WikiText-2 perplexity (PPL ↓), A40 46 GB:

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|-------------|-----------|----------|
| FP16 baseline | 6.24 | 6.85 | 9.92 |
| RTN INT8 | 6.25 (+0.01) | 6.85 (+0.00) | 9.92 (+0.00) |
| GPTQ INT4 | 11.10 (+4.86) | 7.04 (+0.19) | 10.21 (+0.29) |
| AWQ INT4 | 6.77 (+0.53) | 7.13 (+0.28) | 10.36 (+0.44) |

Zero-shot accuracy — 6-task average (LAMBADA, PIQA, WinoGrande, OpenBookQA, RTE, COPA) ↑:

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|-------------|-----------|----------|
| FP16 baseline | 73.22 | 74.13 | 69.57 |
| RTN INT8 | 73.00 (−0.22) | 74.09 (−0.04) | 69.57 (0.00) |
| GPTQ INT4 | 71.65 (−1.57) | 73.39 (−0.74) | 69.42 (−0.15) |
| AWQ INT4 | 70.69 (−2.53) | 73.93 (−0.20) | 69.57 (0.00) |

Throughput at BS=1 (tok/s), dequantize-on-the-fly †:

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|-------------|-----------|----------|
| FP16 | 25.8 | 32.4 | 25.1 |
| RTN INT8 | 2.1 | 2.3 | 2.3 |
| GPTQ INT4 | 1.1 | 1.2 | 1.2 |
| AWQ INT4 | 1.1 | 1.2 | 1.2 |

† With the optional Triton kernel: **8.8× faster at BS=1, 2.8× at BS=4**.

---

## Installation

```bash
pip install pare-quant                   # latest
pip install pare-quant==0.1.0           # pin to specific version
pip install "pare-quant[all]"            # + transformers, datasets, Triton kernel
```

Python ≥ 3.11 · PyTorch ≥ 2.1

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pare import quantize, QuantConfig
from pare.calibration.data import load_wikitext2_calibration

model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
calib     = load_wikitext2_calibration(tokenizer, n_samples=128, seq_len=2048)
# or use your own: a list of tokenized tensors of shape (seq_len,)

# Default is AWQ. Change scheme= to switch methods.
config = QuantConfig(bits=4, scheme="awq", group_size=128)   # ← swap to "gptq", "rtn", "smoothquant"
model  = quantize(model, config, calibration_data=calib, device="cuda")
```

Save and reload:

```python
from pare import save_quantized, load_quantized

save_quantized(model, "qwen25-awq-int4/")
# [pare] Saved 224 quantized layers to qwen25-awq-int4  (3821 MB)

from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
model  = AutoModelForCausalLM.from_config(config)   # architecture only, no weights
model  = load_quantized(model, "qwen25-awq-int4/")
```

---

## Methods

| `scheme=` | Calibration | Quality | When to use |
|-----------|-------------|---------|-------------|
| `"awq"` ★ | Yes | ★★★★ | **Default.** Best robustness across architectures; recommended starting point |
| `"gptq"` | Yes | ★★★★ | Matches AWQ on Qwen2.5-7B; architecture-agnostic — works correctly across pre- and post-norm models |
| `"smoothquant"` | Yes | ★★★★ | INT8 W+A; closest to FP16 PPL; no INT4 |
| `"rtn"` | No | ★★★ | No calibration needed; good baseline or for NF4/FP8 |

★ Default: `QuantConfig()` uses AWQ. AWQ is the strongest INT4 method on Qwen2.5-7B (−0.20 vs FP16 baseline). GPTQ is architecture-agnostic and is recommended when the target model's architecture is uncertain. `act_order=True` may further improve GPTQ quality but has not been benchmarked yet.

All schemes support `bits=4` or `bits=8`. Use `group_size=128` (default) for best INT4 quality.

### Additional options

**`act_order=True`** — Sort quantization by activation magnitude (improves GPTQ quality on modern architectures):
```python
QuantConfig(bits=4, scheme="gptq", group_size=128, act_order=True)
```

**Mixed-precision** — Automatically promote sensitive layers to higher bits:
```python
QuantConfig(bits=4, scheme="awq", sensitive_bits=8, sensitivity_threshold=0.05)
# [pare] 12 of 224 layers promoted to INT8 based on activation-weighted error
```

**NF4** — Normal float 4-bit codebook (QLoRA-compatible base model format):
```python
from pare.core.dtype import QuantDtype
QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
```

**FP8** — 8-bit float for A100/H100:
```python
QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
```

---

## Inference speedup (Triton kernel)

The optional Triton INT4 kernel fuses dequantization into the matmul, avoiding materialising the full FP16 weight matrix. Applies to INT4 schemes (AWQ, GPTQ, RTN). Enable per-layer after quantization:

```python
from pare.layers.linear import QuantizedLinear

for m in model.modules():
    if isinstance(m, QuantizedLinear):
        m.use_kernel = True
```

| Batch size | Without kernel | With kernel | Speedup |
|------------|---------------|-------------|---------|
| 1 (decode) | 2.09 ms/layer | 0.24 ms/layer | **8.8×** |
| 4 | 2.18 ms/layer | 0.78 ms/layer | **2.8×** |
| 16 | 2.66 ms/layer | 3.18 ms/layer | 0.8× |

Requires `pip install triton>=3.0`.

---

## Hardware

| | Minimum |
|-|---------|
| Quantizing a 7B model | 20 GB VRAM (layerwise strategy peaks at ~2 GB) |
| RTN / GPTQ / AWQ / NF4 | Any CUDA GPU |
| SmoothQuant W+A | Any CUDA GPU |
| FP8 | PyTorch ≥ 2.1 (A100 via software; H100 native) |
| Triton kernel | CUDA GPU + `triton ≥ 3.0` |

---

## Citation

```bibtex
@misc{moslem2026pare,
  author = {Moslem, Yasmin},
  title  = {Pare: Production-ready quantization for large language models},
  year   = {2026},
  url    = {https://github.com/TinyAdapt/Pare},
}
```

Apache 2.0
