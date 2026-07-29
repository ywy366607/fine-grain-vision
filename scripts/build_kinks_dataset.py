"""Materialize the synthetic kinks dataset to disk.

Usage:
  python scripts/build_kinks_dataset.py --out data/kinks256 --n_train 6000 --n_val 600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.tasks import make_kinks


class D:
    make_kinks = staticmethod(make_kinks)


def _save_png(path: str, chw: torch.Tensor) -> None:
    """chw float [0,1] -> uint8 PNG."""
    arr = (chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)


def _bbox(msk_flat: np.ndarray, R: int):
    m = msk_flat.reshape(R, R)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return -1, -1, -1, -1, 0
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()), int(m.sum())


def build_split(out_split: str, n: int, res: int, hard_frac: float, hard_tile: int,
                seed: int, k_list: list[int]) -> dict:
    os.makedirs(os.path.join(out_split, "images"), exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    # balanced over k
    per = n // len(k_list)
    extra = n - per * len(k_list)
    ks = []
    for i, k in enumerate(k_list):
        ks.extend([k] * (per + (1 if i < extra else 0)))
    rng.shuffle(ks)
    ks = ks[:n]

    t0 = time.time()
    n_hard = 0
    for i, k in enumerate(ks):
        use_hard = float(rng.random()) < hard_frac
        if use_hard:
            n_hard += 1
        img, lab, msk = D.make_kinks(
            rng, np.array([k]), res=res, hard_tile=hard_tile, hard_frac=1.0 if use_hard else 0.0)
        # make_kinks lab = k - 5; recompute absolute kinks for csv
        chw = img[0]
        flat = msk[0].numpy()
        y0, x0, y1, x1, ink = _bbox(flat, res)
        path = os.path.join(out_split, "images", f"{i:06d}.png")
        _save_png(path, chw)
        rows.append({
            "id": f"{i:06d}",
            "file": f"images/{i:06d}.png",
            "kinks": int(k),
            "class": int(lab[0].item()),
            "hard": int(use_hard),
            "ink": ink,
            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
        })
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  {out_split}: {i+1}/{n}  hard_so_far={n_hard}  "
                  f"{time.time()-t0:.1f}s", flush=True)

    csv_path = os.path.join(out_split, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"n": n, "n_hard": n_hard, "csv": csv_path}


def preview(out: str, res: int, hard_tile: int, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.2))
    for col, k in enumerate(range(5, 11)):
        for row, hard in enumerate([0.0, 1.0]):
            img, lab, msk = D.make_kinks(
                rng, np.array([k]), res=res, hard_tile=hard_tile, hard_frac=hard)
            ax = axes[row, col]
            ax.imshow(np.clip(img[0].permute(1, 2, 0).numpy(), 0, 1), interpolation="nearest")
            mm = msk[0].numpy().reshape(res, res)
            ys, xs = np.where(mm)
            if len(ys):
                ax.add_patch(plt.Rectangle(
                    (xs.min() - 0.5, ys.min() - 0.5),
                    xs.max() - xs.min() + 1, ys.max() - ys.min() + 1,
                    fill=False, ec="yellow", lw=1.0))
            tag = "HARD16" if hard else "free"
            ax.set_title(f"{tag} k={k}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if hard_tile:
                for t in range(0, res + 1, hard_tile):
                    ax.axhline(t - 0.5, color="c", lw=0.2, alpha=0.35)
                    ax.axvline(t - 0.5, color="c", lw=0.2, alpha=0.35)
    axes[0, 0].set_ylabel("free snake")
    axes[1, 0].set_ylabel("hard 16x16")
    fig.suptitle("kinks dataset: free + hard mix (tail-2 isolation)")
    fig.tight_layout()
    prev = os.path.join(out, "preview", "overview.png")
    os.makedirs(os.path.dirname(prev), exist_ok=True)
    fig.savefig(prev, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print("preview →", prev, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "kinks256"))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--hard_frac", type=float, default=0.35,
                    help="fraction of samples packed into hard_tile window")
    ap.add_argument("--hard_tile", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=10)
    args = ap.parse_args()

    k_list = list(range(args.k_min, args.k_max + 1))
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "task": "kinks",
        "res": args.res,
        "k_range": [args.k_min, args.k_max],
        "n_classes": args.k_max - args.k_min + 1,
        "class_map": "class = kinks - k_min",
        "line": "1px pure red SIGNAL[0], axis-aligned",
        "isolation": "9-grid: new red may 8-touch only last 2 path pixels (corner OK)",
        "hard_tile": args.hard_tile,
        "hard_frac": args.hard_frac,
        "hard_def": "entire polyline inside one hard_tile x hard_tile window",
        "generator": "make_kinks (tail-2 clearance)",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed_train": args.seed,
        "seed_val": args.seed + 10_000,
        "note": "on-disk materialization of synthetic data; not ImageNet-style natural images",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Building TRAIN…", flush=True)
    tr = build_split(
        os.path.join(args.out, "train"), args.n_train, args.res,
        args.hard_frac, args.hard_tile, args.seed, k_list)
    print("Building VAL…", flush=True)
    va = build_split(
        os.path.join(args.out, "val"), args.n_val, args.res,
        args.hard_frac, args.hard_tile, args.seed + 10_000, k_list)
    preview(args.out, args.res, args.hard_tile, seed=args.seed + 99)

    summary = {"train": tr, "val": va, "meta": meta}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DONE", args.out, flush=True)
    print(f"  train {tr['n']} (hard≈{tr['n_hard']})  val {va['n']} (hard≈{va['n_hard']})")


if __name__ == "__main__":
    main()
