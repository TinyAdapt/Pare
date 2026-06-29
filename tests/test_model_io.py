"""Tests for save_quantized / load_quantized.

These tests run on CPU with tiny GPT-2 and Qwen2.5-0.5B fixtures.
They verify that a round-trip (quantize → save → load) produces a model
whose output matches the quantized model before saving.

Tests are marked --smoke so they don't run in the default CI suite.
"""

import copy
import tempfile

import pytest
import torch

from pare import QuantConfig, load_quantized, quantize, save_quantized
from pare.eval.perplexity import evaluate_perplexity
from pare.layers.linear import QuantizedLinear


# ---------------------------------------------------------------------------
# GPT-2 round-trips (no model.model.layers — RTN fallback)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestGPT2IO:
    def test_rtn_int8_roundtrip(self, gpt2_model_and_tokenizer, tmp_path):
        """RTN INT8 save/load: PPL matches the quantized model before saving."""
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="rtn", granularity="per_channel")
        quantize(model_q, config)

        # Measure PPL before save
        ppl_before = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=5, device="cpu",
        )

        # Save
        save_quantized(model_q, tmp_path / "gpt2_rtn_int8")

        # Load into a fresh copy of the original model
        model_loaded = copy.deepcopy(model)
        load_quantized(model_loaded, tmp_path / "gpt2_rtn_int8")

        # PPL after load must match before save (within float rounding)
        ppl_after = evaluate_perplexity(
            model_loaded, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=5, device="cpu",
        )
        print(f"\nGPT-2 RTN INT8  PPL before: {ppl_before:.4f}  after: {ppl_after:.4f}")
        assert abs(ppl_after - ppl_before) < 0.01, (
            f"PPL changed after load: {ppl_before:.4f} → {ppl_after:.4f}"
        )

    def test_rtn_int4_roundtrip(self, gpt2_model_and_tokenizer, tmp_path):
        """RTN INT4 save/load round-trip."""
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="rtn", granularity="per_group", group_size=64)
        quantize(model_q, config)

        ppl_before = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=5, device="cpu",
        )

        save_quantized(model_q, tmp_path / "gpt2_rtn_int4")
        model_loaded = copy.deepcopy(model)
        load_quantized(model_loaded, tmp_path / "gpt2_rtn_int4")

        ppl_after = evaluate_perplexity(
            model_loaded, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=5, device="cpu",
        )
        print(f"\nGPT-2 RTN INT4  PPL before: {ppl_before:.4f}  after: {ppl_after:.4f}")
        assert abs(ppl_after - ppl_before) < 0.01

    def test_pare_config_json(self, gpt2_model_and_tokenizer, tmp_path):
        """pare_config.json must list quantized layers with correct metadata."""
        import json

        model, _ = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="rtn", granularity="per_channel")
        quantize(model_q, config)

        save_dir = tmp_path / "gpt2_config_test"
        save_quantized(model_q, save_dir)

        with open(save_dir / "pare_config.json") as f:
            meta = json.load(f)

        assert "quantized_layers" in meta
        assert "pare_version" in meta

        # Check at least one layer has expected fields
        first = next(iter(meta["quantized_layers"].values()))
        assert first["bits"] == 8
        assert first["scheme"] == "rtn"
        assert first["granularity"] == "per_channel"
        assert "in_features" in first
        assert "out_features" in first

    def test_quantized_linear_count(self, gpt2_model_and_tokenizer, tmp_path):
        """Loaded model must have same number of QuantizedLinear layers as saved model."""
        model, _ = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="rtn", granularity="per_channel")
        quantize(model_q, config)

        n_ql_before = sum(1 for _, m in model_q.named_modules() if isinstance(m, QuantizedLinear))
        save_dir = tmp_path / "gpt2_count_test"
        save_quantized(model_q, save_dir)

        model_loaded = copy.deepcopy(model)
        load_quantized(model_loaded, save_dir)

        n_ql_after = sum(1 for _, m in model_loaded.named_modules() if isinstance(m, QuantizedLinear))
        assert n_ql_after == n_ql_before, (
            f"QuantizedLinear count changed: {n_ql_before} → {n_ql_after}"
        )


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B round-trips (layerwise GPTQ + SmoothQuant)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestQwenIO:
    def test_gptq_int4_roundtrip(
        self, qwen_model_and_tokenizer, qwen_calibration_data, tmp_path
    ):
        """Qwen GPTQ INT4 save/load: output logits must be bit-exact after reload."""
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        # Compare forward pass output (not PPL — bit-exact check is stronger)
        input_ids = qwen_calibration_data[0][:, :32]
        with torch.no_grad():
            logits_before = model_q(input_ids).logits

        save_dir = tmp_path / "qwen_gptq_int4"
        save_quantized(model_q, save_dir)

        model_loaded = copy.deepcopy(model)
        load_quantized(model_loaded, save_dir)

        with torch.no_grad():
            logits_after = model_loaded(input_ids).logits

        print(
            f"\nQwen GPTQ INT4 logits max-diff: "
            f"{(logits_after - logits_before).abs().max().item():.2e}"
        )
        torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)

    def test_smoothquant_int8_roundtrip(
        self, qwen_model_and_tokenizer, qwen_calibration_data, tmp_path
    ):
        """Qwen SmoothQuant INT8 save/load: bit-exact logits and quantize_inputs preserved."""
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel")
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        input_ids = qwen_calibration_data[0][:, :32]
        with torch.no_grad():
            logits_before = model_q(input_ids).logits

        save_dir = tmp_path / "qwen_sq_int8"
        save_quantized(model_q, save_dir)

        model_loaded = copy.deepcopy(model)
        load_quantized(model_loaded, save_dir)

        with torch.no_grad():
            logits_after = model_loaded(input_ids).logits

        print(
            f"\nQwen SQ INT8 logits max-diff: "
            f"{(logits_after - logits_before).abs().max().item():.2e}"
        )
        torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)

        # quantize_inputs flag must survive the round-trip
        ql_layers = [m for _, m in model_loaded.named_modules() if isinstance(m, QuantizedLinear)]
        assert all(m.quantize_inputs for m in ql_layers), (
            "quantize_inputs=True not preserved after load"
        )
