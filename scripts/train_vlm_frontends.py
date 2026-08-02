#!/usr/bin/env python3
"""Train/compare A/B/C vision frontends → frozen small LLM (ModelScope, D: cache).

Examples:
  set ML_CACHE_ROOT=D:\\ml_cache
  python scripts/train_vlm_frontends.py --mode smoke --prefer pythia
  python scripts/train_vlm_frontends.py --mode sweep --res 32 --steps 80 --prefer pythia
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# D: caches before any HF/modelscope import side effects
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
os.environ["MODELSCOPE_CACHE"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "modelscope")
os.environ["HF_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "huggingface")
os.environ["TORCH_HOME"] = str(Path(os.environ["ML_CACHE_ROOT"]) / "torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fine_grain.frontends import (  # noqa: E402
    build_frontend,
    patch_token_count,
    suggest_T_grid,
)
from fine_grain.llm_backend import cache_root, load_frozen_lm  # noqa: E402
from fine_grain.vlm_data import make_llava_batch, tokenize_captions  # noqa: E402


class VLMBridge(nn.Module):
    """Frozen LLM + trainable vision frontend; visual tokens prepended to text."""

    def __init__(self, frontend: nn.Module, llm: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.llm = llm
        for p in self.llm.parameters():
            p.requires_grad_(False)

    def forward(self, images, input_ids, attention_mask):
        """Causal LM loss on text tokens only (labels ignore visual prefix)."""
        vis = self.frontend(images)  # FrontendOut
        v = vis.tokens  # [B,T,d]
        if v.dtype != next(self.llm.parameters()).dtype:
            v = v.to(dtype=next(self.llm.parameters()).dtype)
        emb = self.llm.get_input_embeddings()(input_ids)  # [B,L,d]
        inputs_embeds = torch.cat([v, emb], dim=1)
        T = v.shape[1]
        B, L = input_ids.shape
        # attention: visual always kept
        vis_mask = torch.ones(B, T, device=input_ids.device, dtype=attention_mask.dtype)
        attn = torch.cat([vis_mask, attention_mask], dim=1)
        # labels: ignore visual prefix + prompt padding; CE on all text positions
        ignore = torch.full((B, T), -100, device=input_ids.device, dtype=input_ids.dtype)
        labels = torch.cat([ignore, input_ids], dim=1)
        # mask pad in labels
        labels = labels.masked_fill(
            torch.cat([
                torch.zeros(B, T, device=input_ids.device, dtype=torch.bool),
                attention_mask == 0,
            ], dim=1),
            -100,
        )
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        meta = dict(vis.meta)
        meta["T"] = vis.T
        return out.loss, meta


def _trainable(m: nn.Module):
    return [p for p in m.parameters() if p.requires_grad]


@torch.no_grad()
def probe_accuracy(bridge, tokenizer, device, res, n=64, batch=8):
    """Fine-grain probe: teacher-forced token accuracy on synthetic captions."""
    bridge.eval()
    rng = np.random.default_rng(123)
    hits = tot = 0
    left = n
    while left > 0:
        b = min(batch, left)
        data = make_llava_batch(rng, b, res=res, mix=("needle", "kinks"))
        ids, mask = tokenize_captions(tokenizer, data["text"], max_length=32)
        ids, mask = ids.to(device), mask.to(device)
        img = data["image"].to(device)
        vis = bridge.frontend(img)
        v = vis.tokens.to(dtype=next(bridge.llm.parameters()).dtype)
        emb = bridge.llm.get_input_embeddings()(ids)
        inputs = torch.cat([v, emb], dim=1)
        T = v.shape[1]
        attn = torch.cat([
            torch.ones(b, T, device=device, dtype=mask.dtype), mask,
        ], dim=1)
        logits = bridge.llm(inputs_embeds=inputs, attention_mask=attn, use_cache=False).logits
        # predict text positions from logits at visual+text[:-1]
        pred = logits[:, T - 1 : T - 1 + ids.shape[1] - 1].argmax(-1)
        tgt = ids[:, 1:]
        m = mask[:, 1:] > 0
        hits += int((pred[m] == tgt[m]).sum())
        tot += int(m.sum())
        left -= b
    bridge.train()
    return hits / max(tot, 1)


def train_one(kind, T, args, llm, tokenizer, d_llm, device):
    torch.manual_seed(args.seed)
    fe = build_frontend(
        kind, d_llm, res=args.res, T=T, patch=args.patch,
        T_slice=args.T_slice, dim=args.dim, depth=args.depth,
        deslice_topk=args.topk,
    ).to(device)
    bridge = VLMBridge(fe, llm).to(device)
    opt = torch.optim.AdamW(_trainable(bridge), lr=args.lr, weight_decay=0.01)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed + 7)
    hist, slots = [], []
    t0 = time.time()
    bridge.train()
    for step in range(1, args.steps + 1):
        data = make_llava_batch(rng, args.batch, res=args.res)
        img = data["image"].to(device, non_blocking=True)
        ids, mask = tokenize_captions(tokenizer, data["text"], max_length=args.max_len)
        ids, mask = ids.to(device), mask.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, meta = bridge(img, ids, mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "T": int(meta.get("T", T)),
                "kind": kind,
                "seconds": time.time() - t0,
            }
            if "slot" in meta:
                row["slot"] = meta["slot"]
                slots.append(meta["slot"])
            if "slice_slot" in meta:
                row["slot"] = meta["slice_slot"]
                slots.append(meta["slice_slot"])
            if "budget" in meta:
                row["budget"] = meta["budget"]
            hist.append(row)
            print(
                f"  [{kind} T={row['T']}] step {step:4d} loss={row['loss']:.4f} "
                f"meta={ {k: meta[k] for k in meta if k in ('budget','G_le_C','T_patch','T_slice')} }",
                flush=True,
            )
    probe = probe_accuracy(bridge, tokenizer, device, args.res, n=args.probe_n, batch=min(8, args.batch))
    final_loss = hist[-1]["loss"] if hist else float("nan")
    return {
        "kind": kind,
        "T": int(hist[-1]["T"]) if hist else T,
        "final_loss": final_loss,
        "probe_token_acc": float(probe),
        "history": hist,
        "slot_last": slots[-1] if slots else {},
        "meta_example": meta if hist else {},
        "seconds": time.time() - t0,
    }


def run_sweep(args, llm, tokenizer, d_llm, device):
    T_list = args.T_list or suggest_T_grid(args.res, args.patch)
    print(f"T grid (high→low): {T_list}  patch_native={patch_token_count(args.res, args.patch)}", flush=True)
    rows = []
    # A at each T that is valid (T <= native)
    native = patch_token_count(args.res, args.patch)
    for T in T_list:
        if T <= native:
            print(f"\n=== A T={T} ===", flush=True)
            try:
                rows.append(train_one("A", T, args, llm, tokenizer, d_llm, device))
            except Exception as e:
                rows.append({"kind": "A", "T": T, "error": str(e)})
                print(f"  A T={T} FAIL {e}", flush=True)
        print(f"\n=== B T={T} ===", flush=True)
        try:
            rows.append(train_one("B", T, args, llm, tokenizer, d_llm, device))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                rows.append({"kind": "B", "T": T, "error": "OOM", "detail": str(e)[:200]})
                print(f"  B T={T} OOM", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                rows.append({"kind": "B", "T": T, "error": str(e)})
                print(f"  B T={T} FAIL {e}", flush=True)
        # C once per T: T_slice ~ T//2
        print(f"\n=== C T≈{T} (split) ===", flush=True)
        try:
            args_c = args
            # build_frontend C uses T for total target split
            rows.append(train_one("C", T, args_c, llm, tokenizer, d_llm, device))
        except Exception as e:
            rows.append({"kind": "C", "T": T, "error": str(e)})
            print(f"  C T={T} FAIL {e}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "sweep"], default="smoke")
    ap.add_argument("--prefer", default="local", choices=["gemma", "pythia", "local"],
                    help="gemma/pythia via ModelScope on D:; local=tiny offline LM on D:")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--T_list", type=int, nargs="*", default=None)
    ap.add_argument("--T_slice", type=int, default=None)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--probe_n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/published/vlm_abc_table.json")
    ap.add_argument("--conclusion", default="results/published/vlm_abc_conclusion.md")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"cache_root={cache_root()} (must be on D:)", flush=True)
    assert str(cache_root()).upper().startswith("D:"), cache_root()

    print(f"loading LLM prefer={args.prefer} …", flush=True)
    llm, tokenizer, d_llm, note = load_frozen_lm(prefer=args.prefer, device=str(device))
    print(f"backend: {note}", flush=True)

    if args.mode == "smoke":
        args.steps = min(args.steps, 20)
        args.T_list = args.T_list or [patch_token_count(args.res, args.patch), 32]
        # unique preserve order
        seen = set()
        tl = []
        for t in args.T_list:
            if t not in seen:
                seen.add(t)
                tl.append(t)
        args.T_list = tl

    rows = run_sweep(args, llm, tokenizer, d_llm, device)

    # main table
    table = []
    for r in rows:
        if "error" in r:
            table.append({
                "kind": r["kind"], "T": r.get("T"), "status": "error",
                "error": r.get("error"),
            })
        else:
            table.append({
                "kind": r["kind"],
                "T": r["T"],
                "final_loss": r["final_loss"],
                "probe_token_acc": r["probe_token_acc"],
                "slot_last": r.get("slot_last", {}),
                "budget": r.get("meta_example", {}).get("budget"),
                "seconds": r["seconds"],
                "status": "ok",
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "backend": note,
        "cache_root": str(cache_root()),
        "res": args.res,
        "patch": args.patch,
        "steps": args.steps,
        "batch": args.batch,
        "topk": args.topk,
        "T_list": args.T_list or suggest_T_grid(args.res, args.patch),
        "table": table,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # conclusion
    ok = [t for t in table if t.get("status") == "ok"]
    lines = [
        "# VLM frontend A/B/C comparison",
        "",
        f"- Backend: `{note}`",
        f"- Cache: `{cache_root()}` (D: drive)",
        f"- res={args.res} patch={args.patch} steps={args.steps} topk={args.topk} ST=on for B/C",
        "",
        "| kind | T | final_loss | probe_token_acc | notes |",
        "|------|---|------------|-----------------|-------|",
    ]
    for t in table:
        if t.get("status") != "ok":
            lines.append(f"| {t.get('kind')} | {t.get('T')} | — | — | {t.get('error')} |")
        else:
            notes = t.get("budget") or ""
            slot = t.get("slot_last") or {}
            if slot:
                notes += f" PR={slot.get('PR_mass', float('nan')):.1f} r99={slot.get('r99', float('nan')):.1f} sup={slot.get('support', float('nan')):.1f}"
            lines.append(
                f"| {t['kind']} | {t['T']} | {t['final_loss']:.4f} | {t['probe_token_acc']:.3f} | {notes} |"
            )
    lines.extend(["", "## Short conclusion", ""])
    if ok:
        by_kind = {}
        for t in ok:
            by_kind.setdefault(t["kind"], []).append(t)
        # best probe per kind
        best = {k: max(vs, key=lambda x: x["probe_token_acc"]) for k, vs in by_kind.items()}
        order = sorted(best.items(), key=lambda kv: -kv[1]["probe_token_acc"])
        lines.append(
            "- Best probe accuracy by kind (this short run): "
            + ", ".join(f"{k}=T{v['T']} acc={v['probe_token_acc']:.3f}" for k, v in order)
        )
        # T compression for B
        b_rows = sorted([t for t in ok if t["kind"] == "B"], key=lambda x: -x["T"])
        if len(b_rows) >= 2:
            lines.append(
                "- B token compression: "
                + " → ".join(f"T{r['T']}:{r['probe_token_acc']:.3f}" for r in b_rows)
            )
        lines.append(
            "- Slice uses ST + fixed topk write; effective-slot metrics logged when available."
        )
        lines.append(
            "- Short steps / synthetic mix only — not a full LLaVA claim; use for ranking frontends."
        )
    else:
        lines.append("- No successful runs; see JSON errors (OOM/auth).")
    Path(args.conclusion).parent.mkdir(parents=True, exist_ok=True)
    Path(args.conclusion).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== TABLE ===", flush=True)
    print("\n".join(lines), flush=True)
    print(f"json → {args.out}", flush=True)
    print(f"conclusion → {args.conclusion}", flush=True)


if __name__ == "__main__":
    main()
