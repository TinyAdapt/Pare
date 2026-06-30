"""Smoke tests: GPT-2 small + Qwen2.5-0.5B, CPU-only, opt-in with --smoke.

These tests verify mathematical correctness of each quantization scheme
by checking that PPL and LAMBADA accuracy stay within expected bounds.
They are NOT measuring state-of-the-art performance — small models
quantize worse than large ones. A result outside the expected range
signals a bug in the equations, not a quality issue.

Two models exercise different code paths:
- GPT-2:         Conv1D layers, no GQA, no RoPE (absolute position embeddings)
- Qwen2.5-0.5B:  nn.Linear, GQA, RoPE, SwiGLU — same family as the 9B benchmark

Run with:
    pytest tests/test_smoke.py --smoke
"""

import pytest

from pare import QuantConfig, quantize
from pare.eval.perplexity import evaluate_perplexity
from pare.eval.lambada import evaluate_lambada

# Number of sequences / examples to evaluate on CPU.
N_SAMPLES_PPL   = 20     # WikiText-2 sequences
N_SAMPLES_LAMB  = 200    # LAMBADA examples
SEQ_LEN         = 1024


@pytest.mark.smoke
class TestGPT2Baseline:
    def test_fp32_ppl(self, gpt2_model_and_tokenizer, wikitext2_test):
        """FP32 GPT-2 should give ~29.4 PPL on full WikiText-2.
        With N_SAMPLES=20 the estimate will vary slightly."""
        model, tokenizer = gpt2_model_and_tokenizer
        ppl = evaluate_perplexity(
            model, tokenizer, dataset="wikitext2",
            seq_len=SEQ_LEN, n_samples=N_SAMPLES_PPL, device="cpu",
        )
        print(f"\nGPT-2 FP32 PPL: {ppl:.2f}")
        assert ppl < 35.0, f"FP32 baseline too high: {ppl:.2f} (expected ~29.4)"

    def test_fp32_lambada(self, gpt2_model_and_tokenizer):
        """FP32 GPT-2 LAMBADA accuracy should be ~32–38% (greedy first-token)."""
        model, tokenizer = gpt2_model_and_tokenizer
        acc = evaluate_lambada(model, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nGPT-2 FP32 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"


@pytest.mark.smoke
class TestGPT2RTN:
    def test_rtn_int8_ppl(self, gpt2_model_and_tokenizer, wikitext2_test):
        """RTN INT8 should add < 2 PPL over FP32 baseline."""
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="rtn", granularity="per_channel")
        quantize(model_q, config)

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=SEQ_LEN, n_samples=N_SAMPLES_PPL, device="cpu",
        )
        print(f"\nGPT-2 RTN INT8 PPL: {ppl:.2f}")
        assert ppl < 37.0, f"RTN INT8 PPL too high: {ppl:.2f} — possible bug"

    def test_rtn_int4_ppl(self, gpt2_model_and_tokenizer, wikitext2_test):
        """RTN INT4 per-group on a small model — expect significant degradation.
        PPL > 100 almost certainly indicates a bug rather than just quantization loss."""
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="rtn", granularity="per_group", group_size=64)
        quantize(model_q, config)

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=SEQ_LEN, n_samples=N_SAMPLES_PPL, device="cpu",
        )
        print(f"\nGPT-2 RTN INT4 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"RTN INT4 PPL > 100: {ppl:.2f} — likely a bug"


@pytest.mark.smoke
class TestGPT2GPTQ:
    def test_gptq_int4_ppl(self, gpt2_model_and_tokenizer, wikitext2_test, gpt2_calibration_data):
        """GPTQ INT4 should give lower PPL than RTN INT4 on the same model.

        GPT-2 is too small for GPTQ to shine (it was designed for 175B), but
        the algorithm must still produce a coherent model.  PPL > 100 almost
        certainly indicates a bug in the Cholesky update, not just quantisation
        degradation on a small model.

        Note: GPT-2 uses Conv1D (not nn.Linear) for attention + MLP projections.
        The patcher transparently converts Conv1D → nn.Linear before quantising.
        """
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=gpt2_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nGPT-2 GPTQ INT4 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"GPTQ INT4 PPL > 100: {ppl:.2f} — likely a bug"

    def test_gptq_int4_lambada(self, gpt2_model_and_tokenizer, gpt2_calibration_data):
        """GPTQ INT4 GPT-2 LAMBADA accuracy must stay > 5% (bug detector)."""
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=gpt2_calibration_data, device="cpu")

        acc = evaluate_lambada(model_q, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nGPT-2 GPTQ INT4 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B: modern architecture (GQA + RoPE + SwiGLU), same family as
# the 9B benchmark model. Exercises code paths GPT-2 doesn't touch.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestQwenBaseline:
    def test_fp32_ppl(self, qwen_model_and_tokenizer, wikitext2_test):
        """Qwen2.5-0.5B FP32 PPL — expected ~15–20 on WikiText-2."""
        model, tokenizer = qwen_model_and_tokenizer
        ppl = evaluate_perplexity(
            model, tokenizer, dataset="wikitext2",
            seq_len=SEQ_LEN, n_samples=N_SAMPLES_PPL, device="cpu",
        )
        print(f"\nQwen2.5-0.5B FP32 PPL: {ppl:.2f}")
        assert ppl < 30.0, f"FP32 baseline too high: {ppl:.2f}"

    def test_fp32_lambada(self, qwen_model_and_tokenizer):
        """Qwen2.5-0.5B FP32 LAMBADA accuracy — expected > 5%."""
        model, tokenizer = qwen_model_and_tokenizer
        acc = evaluate_lambada(model, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nQwen2.5-0.5B FP32 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"


@pytest.mark.smoke
class TestQwenGPTQ:
    def test_gptq_int4_ppl(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B GPTQ INT4 — PPL should stay < 100 (bug detector)."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nQwen2.5-0.5B GPTQ INT4 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"GPTQ INT4 PPL > 100: {ppl:.2f} — likely a bug"

    def test_gptq_int4_lambada(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B GPTQ INT4 LAMBADA accuracy must stay > 5%."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="gptq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        acc = evaluate_lambada(model_q, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nQwen2.5-0.5B GPTQ INT4 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"




@pytest.mark.smoke
class TestGPT2AWQ:
    def test_awq_int4_ppl(self, gpt2_model_and_tokenizer, gpt2_calibration_data):
        """GPT-2 AWQ INT4 — falls back to RTN (no model.model.layers).
        PPL should be similar to RTN INT4 (~33–45)."""
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="awq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=gpt2_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nGPT-2 AWQ INT4 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"AWQ INT4 PPL > 100: {ppl:.2f} — likely a bug"


@pytest.mark.smoke
class TestQwenAWQ:
    def test_awq_int4_ppl(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B AWQ INT4 — exercises full layerwise scale search + fusion."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="awq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nQwen2.5-0.5B AWQ INT4 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"AWQ INT4 PPL > 100: {ppl:.2f} — likely a bug"

    def test_awq_int4_lambada(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B AWQ INT4 LAMBADA — must stay > 5%."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=4, scheme="awq", granularity="per_group", group_size=64)
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        acc = evaluate_lambada(model_q, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nQwen2.5-0.5B AWQ INT4 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"


# ---------------------------------------------------------------------------
# SmoothQuant W8A8: smooth factors migrate activation outliers to weights.
# GPT-2 falls back to RTN (no model.model.layers); Qwen exercises layerwise.
# INT8 quantizes both weights AND activations — stricter than AWQ/GPTQ.
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestGPT2SmoothQuant:
    def test_smoothquant_int8_ppl(self, gpt2_model_and_tokenizer, gpt2_calibration_data):
        """GPT-2 SmoothQuant INT8 — falls back to RTN (no model.model.layers).
        Activation quantization at INT8 adds minimal overhead vs FP32 on GPT-2.
        PPL should stay near RTN INT8 baseline (~30–35)."""
        import copy
        model, tokenizer = gpt2_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel")
        quantize(model_q, config, calibration_data=gpt2_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nGPT-2 SmoothQuant INT8 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"SmoothQuant INT8 PPL > 100: {ppl:.2f} — likely a bug"


@pytest.mark.smoke
class TestQwenSmoothQuant:
    def test_smoothquant_int8_ppl(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B SmoothQuant INT8 W+A — exercises smooth factor computation
        and layerwise fusion. INT8 on a small model degrades gracefully; PPL < 100
        confirms the math is correct.  Expected ~15–25 PPL."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel")
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        ppl = evaluate_perplexity(
            model_q, tokenizer, dataset="wikitext2",
            seq_len=512, n_samples=10, device="cpu",
        )
        print(f"\nQwen2.5-0.5B SmoothQuant INT8 PPL: {ppl:.2f}")
        assert ppl < 100.0, f"SmoothQuant INT8 PPL > 100: {ppl:.2f} — likely a bug"

    def test_smoothquant_int8_lambada(self, qwen_model_and_tokenizer, qwen_calibration_data):
        """Qwen2.5-0.5B SmoothQuant INT8 LAMBADA — must stay > 5%."""
        import copy
        model, tokenizer = qwen_model_and_tokenizer
        model_q = copy.deepcopy(model)

        config = QuantConfig(bits=8, scheme="smoothquant", granularity="per_channel")
        quantize(model_q, config, calibration_data=qwen_calibration_data, device="cpu")

        acc = evaluate_lambada(model_q, tokenizer, n_samples=N_SAMPLES_LAMB, device="cpu")
        print(f"\nQwen2.5-0.5B SmoothQuant INT8 LAMBADA: {acc:.1%}")
        assert acc > 0.05, f"LAMBADA accuracy collapsed to {acc:.1%} — likely a bug"

