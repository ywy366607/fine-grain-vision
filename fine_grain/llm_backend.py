"""Load small causal LMs via ModelScope into D:\\ml_cache (never C:).

Order:
  1) ModelScope snapshot (ignore onnx/tflite/flax/tf junk)
  2) Local tiny GPT-2-style LM written under D:\\ml_cache\\local_tinylm

Prefer: Qwen2.5-0.5B / Gemma-3-270M / GPT2 / Pythia ids when present on MS.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Tuple

# Force non-C caches before importing download stacks
_D_CACHE = Path(os.environ.get("ML_CACHE_ROOT", r"D:\ml_cache"))
_D_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MODELSCOPE_CACHE"] = str(_D_CACHE / "modelscope")
os.environ["HF_HOME"] = str(_D_CACHE / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_D_CACHE / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(_D_CACHE / "huggingface" / "transformers")
os.environ["TORCH_HOME"] = str(_D_CACHE / "torch")

for _sub in (
    os.environ["MODELSCOPE_CACHE"],
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["TORCH_HOME"],
):
    Path(_sub).mkdir(parents=True, exist_ok=True)

_IGNORE = [
    r".*\.tflite$",
    r".*\.onnx$",
    r".*\.msgpack$",
    r".*\.h5$",
    r".*\.ot$",
    r".*\.flax.*",
    r".*rust_model.*",
    r".*tf_model.*",
    r".*flax_model.*",
]

# Prefer short ids known to work on ModelScope CN; HF-style mapped when possible
_MS_MAP = {
    "google/gemma-3-270m": "LLM-Research/gemma-3-270m",
    "google/gemma-3-270m-it": "LLM-Research/gemma-3-270m-it",
    "EleutherAI/pythia-160m": "AI-ModelScope/pythia-160m",
    "EleutherAI/pythia-70m": "AI-ModelScope/pythia-70m",
    "gpt2": "AI-ModelScope/gpt2",
    "Qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}


def cache_root() -> Path:
    return _D_CACHE


def snapshot_modelscope(model_id: str) -> str:
    """Download pytorch-relevant files only to D:\\ml_cache\\modelscope\\hub."""
    from modelscope import snapshot_download

    ms_id = _MS_MAP.get(model_id, model_id)
    cache_dir = str(_D_CACHE / "modelscope" / "hub")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        ms_id,
        cache_dir=cache_dir,
        ignore_file_pattern=_IGNORE,
    )


def _ensure_local_tinylm() -> str:
    """Create a tiny GPT-2-style causal LM on D: (no network). Vocab==model size."""
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    root = _D_CACHE / "local_tinylm"
    root.mkdir(parents=True, exist_ok=True)
    has_w = (root / "model.safetensors").is_file() or (root / "pytorch_model.bin").is_file()
    if (root / "config.json").is_file() and has_w and (root / "tokenizer.json").is_file():
        return str(root)

    V = 512
    specials = ["[PAD]", "[EOS]", "[UNK]"]
    words = (
        "The small square is red green blue yellow polyline has corners "
        "angle degrees A synthetic scene with colored shapes An image of a "
        "line the and to for in on at 15 30 45 60 90 5 6 7 8 9 10"
    ).split()
    vocab = {}
    for s in specials:
        vocab[s] = len(vocab)
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    i = 0
    while len(vocab) < V:
        k = f"#{i}"
        if k not in vocab:
            vocab[k] = len(vocab)
        i += 1

    tok_m = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_m.pre_tokenizer = Whitespace()
    tok_m.save(str(root / "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    hf_tok.model_max_length = 128
    hf_tok.save_pretrained(str(root))

    cfg = GPT2Config(
        vocab_size=V,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=512,
        bos_token_id=vocab["[EOS]"],
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    GPT2LMHeadModel(cfg).save_pretrained(str(root))
    return str(root)


def _load_from_path(local: str, device: str, dtype: Any) -> Tuple[Any, Any, int, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or getattr(tok, "unk_token", "[PAD]")
    model = AutoModelForCausalLM.from_pretrained(
        local,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # vocab size mismatch guard for tinylm
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_llm = int(model.config.hidden_size)
    return model, tok, d_llm, local


def _local_manual_dirs() -> List[Path]:
    """Local model dirs on D: (curl copies + HF hub snapshots)."""
    found: List[Path] = []
    root = _D_CACHE / "huggingface" / "models"
    if root.is_dir():
        found.extend(
            p for p in root.iterdir() if p.is_dir() and (p / "config.json").is_file()
        )
    # HF hub cache layout: hub/models--org--name/snapshots/<hash>/
    hub = _D_CACHE / "huggingface" / "hub"
    if hub.is_dir():
        for repo in hub.glob("models--*"):
            snaps = repo / "snapshots"
            if not snaps.is_dir():
                continue
            # newest snapshot
            candidates = sorted(
                [s for s in snaps.iterdir() if s.is_dir() and (s / "config.json").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                found.append(candidates[0])
    return found


def load_frozen_lm(
    prefer: str = "gemma",
    device: str = "cuda",
    dtype: Any = None,
) -> Tuple[Any, Any, int, str]:
    """Return (model, tokenizer, d_llm, backend_note). LLM frozen. Cache on D:."""
    import torch

    # Gemma-3-270M NaNs under fp16 on some consumer GPUs; default fp32 when prefer=gemma
    # (270M weights ~1GB — still fits 4GB). Others may use fp16.
    if dtype is None:
        if prefer == "gemma":
            dtype = torch.float32
        else:
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    errors: List[str] = []

    # 1) Local downloads on D: (curl / HF hub snapshots)
    manuals = _local_manual_dirs()
    order = []
    if prefer == "gemma":
        order = ["gemma-3-270m", "gemma", "pythia-160m", "pythia-70m"]
    elif prefer == "pythia":
        order = ["pythia-160m", "pythia-70m", "gemma"]
    elif prefer == "local":
        order = []
    else:
        order = [prefer, "pythia-160m", "gemma"]
    ranked: List[Path] = []
    for key in order:
        for p in manuals:
            if key.lower() in p.name.lower() and p not in ranked:
                ranked.append(p)
    for p in manuals:
        if p not in ranked:
            ranked.append(p)
    for p in ranked:
        try:
            if not str(p).upper().startswith("D:"):
                raise RuntimeError(f"refusing non-D path {p}")
            # force fp32 for gemma* paths even if prefer was generic
            use_dtype = torch.float32 if "gemma" in p.name.lower() else dtype
            model, tok, d_llm, path = _load_from_path(str(p), device, use_dtype)
            note = (
                f"local-D path={path} d_llm={d_llm} dtype={use_dtype} cache={_D_CACHE}"
            )
            return model, tok, d_llm, note
        except Exception as e:
            errors.append(f"manual:{p.name}: {type(e).__name__}: {e}")

    # 2) ModelScope
    if prefer == "gemma":
        candidates = ["google/gemma-3-270m", "EleutherAI/pythia-160m", "gpt2"]
    elif prefer == "pythia":
        candidates = ["EleutherAI/pythia-160m", "gpt2"]
    elif prefer == "local":
        candidates = []
    else:
        candidates = [prefer, "EleutherAI/pythia-160m"]
    for mid in candidates:
        try:
            local = snapshot_modelscope(mid)
            if str(local).upper().startswith("C:"):
                raise RuntimeError(f"refusing C: path {local}")
            model, tok, d_llm, path = _load_from_path(local, device, dtype)
            note = f"modelscope:{mid} path={path} d_llm={d_llm} cache={_D_CACHE}"
            return model, tok, d_llm, note
        except Exception as e:
            errors.append(f"{mid}: {type(e).__name__}: {e}")

    # 3) Offline tinylm on D:
    try:
        local = _ensure_local_tinylm()
        assert str(local).upper().startswith("D:")
        model, tok, d_llm, path = _load_from_path(local, device, dtype)
        note = (
            f"local_tinylm path={path} d_llm={d_llm} cache={_D_CACHE} "
            f"(prior fails: {len(errors)})"
        )
        return model, tok, d_llm, note
    except Exception as e:
        errors.append(f"local_tinylm: {type(e).__name__}: {e}")
        raise RuntimeError(
            "Failed to load LM on D: (hf-mirror local / ModelScope / tinylm):\n"
            + "\n".join(errors)
        )
