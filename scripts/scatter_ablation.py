#!/usr/bin/env python3
"""Short train-from-scratch line-recon ablation: soft deslice vs sparse write vs gate.

PRIMARY comparison is soft write vs top-k sparse write (same pool/read soft).
Gate is Qwen residual-stream / post-SDPA optional; recurrence is fallback only.

Example:
  python scripts/scatter_ablation.py --steps 200 --res 64 --rgb --amp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fine_grain as D
from scripts.line_recon import (  # noqa: E402
    SliceSeg,
    dice_loss_with_logits,
    make_batch,
    metrics,
)


@torch.no_grad()
def metrics_fp(logits, target, thr=0.5):
    m = metrics(logits, target, thr=thr)
    p = (logits.sigmoid() > thr).float()
    fp = ((p > 0.5) & (target < 0.5)).float().sum(dim=(1, 2)).mean().item()
    return {**m, "fp_mean": float(fp)}


def support_of(model):
    mix = model.inner.blocks[0].mix
    w = getattr(mix, "last_w_write", None)
    if w is None:
        return float("nan")
    return float(D.deslice_support_size(w))


def build_named(name, dim, depth, slice_num):
    if name in D.ARMS and D.ARMS[name].get("kind") == "slice":
        spec = dict(D.ARMS[name])
        m = SliceSeg(
            dim=dim, depth=depth, slice_num=slice_num,
            local=bool(spec.get("local", False)),
            nog=bool(spec.get("nog", False)),
        )
        D.apply_slice_flags(m.inner, spec)
        return m
    raise ValueError(name)


def run_one(name, args, device):
    torch.manual_seed(args.seed)
    model = build_named(name, args.dim, args.depth, args.slice_num).to(device)
    opt = D._make_optimizer(model.parameters(), args.lr, device.type == "cuda")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_w = torch.tensor([args.pos_weight], device=device)
    k_list = list(range(args.k_min, args.k_max + 1))
    rng = np.random.default_rng(1000 + args.seed)
    hist = []
    t0 = time.time()

    def evaluate(step):
        model.eval()
        rng_e = np.random.default_rng(90_000 + step)
        dices, ious, fps, sups = [], [], [], []
        n_left = args.eval_n
        while n_left > 0:
            b = min(args.eval_batch, n_left)
            x, m = make_batch(
                rng_e, b, args.res, args.hard_frac, args.hard_tile, k_list, rgb=args.rgb,
            )
            x, m = x.to(device), m.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            met = metrics_fp(logits.float(), m)
            dices.append(met["dice"])
            ious.append(met["iou"])
            fps.append(met["fp_mean"])
            sups.append(support_of(model))
            n_left -= b
        model.train()
        return dict(
            step=step,
            dice=float(np.mean(dices)),
            iou=float(np.mean(ious)),
            fp_mean=float(np.mean(fps)),
            support=float(np.nanmean(sups)),
            seconds=time.time() - t0,
        )

    st0 = evaluate(0)
    hist.append(st0)
    print(
        f"  [{name}] step 0  dice={st0['dice']:.3f} fp={st0['fp_mean']:.1f} "
        f"support={st0['support']:.2f}",
        flush=True,
    )
    model.train()
    for step in range(1, args.steps + 1):
        x, m = make_batch(
            rng, args.batch, args.res, args.hard_frac, args.hard_tile, k_list, rgb=args.rgb,
        )
        x, m = x.to(device), m.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, m, pos_weight=pos_w)
            loss = loss + args.dice_w * dice_loss_with_logits(logits.float(), m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % args.probe_every == 0 or step == args.steps:
            st = evaluate(step)
            st["loss"] = float(loss.item())
            hist.append(st)
            print(
                f"  [{name}] step {step:4d}  dice={st['dice']:.3f} iou={st['iou']:.3f} "
                f"fp={st['fp_mean']:.1f} support={st['support']:.2f} loss={st['loss']:.3f}",
                flush=True,
            )
    return dict(arm=name, history=hist, final=hist[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arms",
        default="slice_loc_nogumbel,slice_loc_nogumbel_topk2,"
                "slice_loc_nogumbel_gate,slice_loc_nogumbel_st_topk2",
    )
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--probe_every", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--hard_frac", type=float, default=0.35)
    ap.add_argument("--hard_tile", type=int, default=16)
    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--slice_num", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos_weight", type=float, default=40.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--out", default="results/published/scatter_ablation.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"scatter-ablation | res={args.res} steps={args.steps} rgb={int(args.rgb)} "
        f"arms={args.arms} device={device}",
        flush=True,
    )
    print(
        "PRIMARY: soft write vs top-k sparse write (soft pool kept). "
        "Gate=Qwen residual/SDPA placement. Recur arms are fallback only.",
        flush=True,
    )

    results = []
    for name in args.arms.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        results.append(run_one(name, args, device))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {
        "task": "scatter_ablation",
        "rgb": bool(args.rgb),
        "res": args.res,
        "steps": args.steps,
        "batch": args.batch,
        "hard_frac": args.hard_frac,
        "arms": results,
        "note": (
            "Primary soft-scatter fix is deslice top-k/threshold on WRITE only. "
            "Qwen gate: post-SDPA on slice tokens + optional residual-stream res_gate. "
            "Recurrence (recur_T) is documented fallback, not primary."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===", flush=True)
    print(f"{'arm':<36} {'dice':>6} {'iou':>6} {'fp':>7} {'sup':>5}", flush=True)
    for r in results:
        f = r["final"]
        print(
            f"{r['arm']:<36} {f['dice']:6.3f} {f['iou']:6.3f} "
            f"{f['fp_mean']:7.1f} {f['support']:5.1f}",
            flush=True,
        )
    print(f"json → {args.out}", flush=True)


if __name__ == "__main__":
    main()
