# Pare

Quantize any LLM in one line. Switch between GPTQ, AWQ, SmoothQuant, and RTN by changing a config field — same API, same model, same output format.

---

## Pick your trade-off

WikiText-2 perplexity (PPL ↓), A40 46 GB:

| Method | Llama-2-7B | Llama-3-8B | Qwen2.5-7B |
|--------|-----------|-----------|-----------|
| FP16 baseline | 5.47 | 6.14 | 6.85 |
| RTN INT8 | 5.48 (+0.01) | 6.14 (+0.01) | 6.85 (+0.00) |
| SmoothQuant INT8 | 5.58 (+0.11) | 6.25 (+0.11) | 6.96 (+0.11) |
| AWQ INT4 | 5.67 (+0.20) | 6.67 (+0.53) | 7.13 (+0.28) |
| GPTQ INT4 | 5.74 (+0.27) | 8.75 (+2.61) | 7.04 (+0.19) |

Throughput on Llama-2-7B (BS=1): FP16 33 tok/s · RTN/SmoothQuant ~2.3 tok/s · AWQ/GPTQ ~1.2 tok/s †

† Dequantize-on-the-fly. With the optional Triton kernel: **8.8× faster at BS=1, 2.8× at BS=4**.

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pare import quantize, QuantConfig
from pare.calibration.data import load_wikitext2_calibration

model     = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
calib     = load_wikitext2_calibration(tokenizer, n_samples=128, seq_len=2048)
# or use your own: a list of tokenized tensors of shape (seq_len,)

# Default is AWQ. Change scheme= to switch methods — everything else stays the same.
config = QuantConfig(bits=4, scheme="awq", group_size=128)   # ← swap to "gptq", "rtn", "smoothquant"
model  = quantize(model, config, calibration_data=calib, device="cuda")
```

Save and reload:

```python
from pare import save_quantized, load_quantized

save_quantized(model, "llama2-awq-int4/")
# [pare] Saved 224 quantized layers to llama2-awq-int4  (3821 MB)

from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
model  = AutoModelForCausalLM.from_config(config)   # architecture only, no weights
model  = load_quantized(model, "llama2-awq-int4/")
```

---

## Installation

```bash
pip install pare-quant                   # core
pip install "pare-quant[all]"            # + transformers, datasets, Triton kernel
```

Python ≥ 3.11 · PyTorch ≥ 2.1

---

## Methods

| `scheme=` | Calibration | Quality | When to use |
|-----------|-------------|---------|-------------|
| `"awq"` ★ | Yes | ★★★★ | **Default.** Best robustness across architectures; recommended starting point |
| `"gptq"` | Yes | ★★★★★ | Highest quality when used with `act_order=True`; can underperform AWQ without it |
| `"smoothquant"` | Yes | ★★★★ | INT8 W+A; closest to FP16 PPL; no INT4 |
| `"rtn"` | No | ★★★ | No calibration needed; good baseline or for NF4/FP8 |

★ Default: `QuantConfig()` uses AWQ. AWQ consistently outperforms GPTQ (without `act_order=True`) on modern architectures — on Llama-3-8B the gap is 6.67 vs 8.75 PPL. GPTQ with `act_order=True` is the highest-quality option but requires more tuning.

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
  title  = {Pare: Production-ready quantization for large language and multimodal models},
  year   = {2026},
  url    = {https://github.com/TinyAdapt/Pare},
}
```

Apache 2.0
