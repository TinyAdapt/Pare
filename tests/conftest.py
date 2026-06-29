"""Shared pytest fixtures and configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Run smoke tests (GPT-2 + WikiText-2, requires internet, CPU-only)",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires CUDA)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: opt-in end-to-end CPU tests")
    config.addinivalue_line("markers", "gpu: tests requiring CUDA")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--smoke"):
        skip_smoke = pytest.mark.skip(reason="pass --smoke to run")
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip_smoke)

    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="pass --gpu to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpt2_model_and_tokenizer():
    """Load GPT-2 small once per test session (cached after first download)."""
    pytest.importorskip("transformers", reason="transformers required for smoke tests")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="auto")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def wikitext2_test():
    """Load WikiText-2 test split (~2MB, cached after first download)."""
    pytest.importorskip("datasets", reason="datasets required for smoke tests")
    from datasets import load_dataset
    return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")


@pytest.fixture(scope="session")
def gpt2_calibration_data(gpt2_model_and_tokenizer, wikitext2_test):
    """Prepare 16 calibration sequences for GPTQ (each 512 tokens).

    Uses the WikiText-2 test text tokenized with GPT-2's tokenizer.
    For a production run you would use the training split; for smoke
    tests any text produces a valid (if noisy) Hessian estimate.
    """
    import torch
    _, tokenizer = gpt2_model_and_tokenizer
    text = "\n\n".join(wikitext2_test["text"])
    tokens = tokenizer.encode(text, return_tensors="pt")  # [1, n_tokens]

    seq_len = 512
    n_calib = 16
    return [tokens[:, i * seq_len : (i + 1) * seq_len] for i in range(n_calib)]


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B fixtures (modern architecture: GQA + RoPE + SwiGLU)
# ---------------------------------------------------------------------------

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def qwen_model_and_tokenizer():
    """Load Qwen2.5-0.5B once per session (~1 GB, CPU-only)."""
    pytest.importorskip("transformers", reason="transformers required for smoke tests")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype="auto")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def qwen_calibration_data(qwen_model_and_tokenizer, wikitext2_test):
    """8 calibration sequences × 512 tokens for Qwen2.5-0.5B GPTQ."""
    import torch
    _, tokenizer = qwen_model_and_tokenizer
    text = "\n\n".join(wikitext2_test["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids

    seq_len = 512
    n_calib = 8
    return [tokens[:, i * seq_len : (i + 1) * seq_len] for i in range(n_calib)]
