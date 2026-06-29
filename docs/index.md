# Pare

**Production-ready quantization for large language and multimodal models.**

`pare` is a modular, research-grade library for post-training quantization (PTQ) of LLMs. It implements RTN, GPTQ, AWQ, and SmoothQuant under a unified API, with built-in evaluation and optional Triton kernels for real inference speedup.

## Features

- **Algorithms**: RTN, GPTQ, AWQ, SmoothQuant
- **Formats**: INT4, INT8, FP8 (E4M3/E5M2), NF4
- **Granularity**: per-tensor, per-channel, per-group
- **Hardware**: tested on A40, A100 (FP8), RTX 6000 Pro
- **HuggingFace compatible**: load any `transformers` model, save quantized weights via `safetensors`

## Quick example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pare import QuantConfig, quantize
from pare.calibration.data import load_wikitext2_calibration

model     = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
calib     = load_wikitext2_calibration(tokenizer, n_samples=128, seq_len=2048)

config    = QuantConfig(bits=4, scheme="awq", group_size=128)
model     = quantize(model, config, calibration_data=calib, device="cuda")
```

## Installation

```bash
pip install pare-quant
pip install "pare-quant[all]"   # + transformers, datasets, Triton kernel
```

See the [README](https://github.com/TinyAdapt/Pare) for the full benchmark table and usage guide.
