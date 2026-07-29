"""Training helpers, collapse probes, and data loading."""
from __future__ import annotations

import math
import os

import numpy as np
import torch

def _first_mix(model):
    """Unwrap torch.compile / nested modules to the first AdaTempSlice."""
    root = getattr(model, "_orig_mod", model)
    mix = root.blocks[0].mix
    return getattr(mix, "_orig_mod", mix)


def pr_obj(model, mask):
    """P5': how many slices the label-bearing pixels actually spread over.
    ~1 => the object collapsed into a single slice (shape-blind); >>1 => a parts
    decomposition formed. Averaged over heads and over the first block's assignment."""
    w = getattr(_first_mix(model), "last_w", None)
    if w is None:
        return float("nan")
    m = mask.to(w.device).float()[:, None, :, None]              # [B,1,N,1]
    p = (w * m).sum(2) / m.sum(2).clamp(min=1)                   # [B,H,G]
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-8)
    return float((1.0 / p.pow(2).sum(-1)).mean())


def collapse_stats(model):
    """Slice-collapse panel (block-0, eval assignment after a forward).

    Transolver++ Ada-Temp targets "slice collapse": at large N the soft assignment
    goes near-uniform and slice tokens become homogeneous. We measure BOTH sides:

      PR_mass   effective #slices in the *average* assignment mass over all points.
                ~1 => all mass on one slice (routing collapse / dead slices)
                ~G => mass spread uniformly across slices
      H_mass    PR_mass's entropy cousin, normalized to [0,1] by log(G).
                ~0 delta-collapse; ~1 uniform (the other collapse mode)
      H_point   mean per-point assignment entropy / log(G).
                low => hard routing (can be specialization OR winner-take-all)
      cos_tok   mean pairwise cosine of the G mass-normalized slice tokens.
                ~1 => tokens are copies (representation collapse); ~0 diverse
      r99       smallest k s.t. top-k slices hold 99% of mean mass (per head, avg).
                r99≪G => most slices idle

    PR_obj (elsewhere) is object-conditional and does NOT measure global collapse;
    high PR_obj with chance accuracy already falsified "dispersion = shape".
    """
    mix = _first_mix(model)
    w = getattr(mix, "last_w", None)       # [B,H,N,G]
    tok = getattr(mix, "last_tok", None)   # [B,H,G,Dh]
    if w is None:
        return {k: float("nan") for k in
                ("PR_mass", "H_mass", "H_point", "cos_tok", "r99")}
    B, H, N, G = w.shape
    # --- mass over points: how is total assignment budget spread across slices?
    mass = w.sum(2)                                         # [B,H,G]
    p = mass / mass.sum(-1, keepdim=True).clamp(min=1e-8)
    pr_mass = float((1.0 / p.pow(2).sum(-1)).mean())
    h_mass = float((-(p * (p + 1e-8).log()).sum(-1) / math.log(G)).mean())
    # r99: cumulative mass after sorting
    ps, _ = p.sort(dim=-1, descending=True)
    cume = ps.cumsum(-1)
    # first index where cume >= 0.99, 1-based count
    hit = (cume >= 0.99)
    # if never hits (num error), G
    idx = hit.float().argmax(dim=-1)                        # 0 if none True...
    none = ~hit.any(dim=-1)
    r99 = (idx + 1).float()
    r99[none] = float(G)
    r99 = float(r99.mean())
    # --- per-point softness
    h_point = float((-(w * (w + 1e-8).log()).sum(-1) / math.log(G)).mean())
    # --- slice-token geometry
    cos_tok = float("nan")
    if tok is not None and G >= 2:
        t = tok / tok.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [B,H,G,D]
        sim = torch.matmul(t, t.transpose(-1, -2))                   # [B,H,G,G]
        eye = torch.eye(G, device=sim.device, dtype=torch.bool)
        cos_tok = float(sim.masked_select(~eye.view(1, 1, G, G)).mean())
    return dict(PR_mass=pr_mass, H_mass=h_mass, H_point=h_point,
                cos_tok=cos_tok, r99=r99)


def _build_train_pool(make, rng, sizes, res, mult, pool_n, batch):
    """Amortize CPU synthetic rendering: build a fixed on-host pool once per arm/seed.
    Training then only indexes + H2D (or stays on device). Task distribution unchanged
    (same generator, uniform over task sizes); only the sampling buffer is finite."""
    imgs, labs = [], []
    left = pool_n
    while left > 0:
        b = min(batch, left)
        s = sizes[rng.integers(0, len(sizes), b)]
        img, lab, _ = make(rng, s, res, mult)
        imgs.append(img)
        labs.append(lab)
        left -= b
    return torch.cat(imgs, 0), torch.cat(labs, 0)


def load_kinks_folder(split_dir: str):
    """Load materialised kinks split from build_kinks_dataset.py (labels.csv + PNGs)."""
    import csv
    from PIL import Image
    csv_path = os.path.join(split_dir, "labels.csv")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    imgs, labs = [], []
    for r in rows:
        p = os.path.join(split_dir, r["file"])
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        labs.append(int(r["class"]))
    return torch.stack(imgs, 0), torch.tensor(labs, dtype=torch.int64)


def _make_optimizer(params, lr, use_cuda):
    # fused AdamW is a free win on CUDA when available (PyTorch 2.x).
    kwargs = dict(lr=lr, weight_decay=0.01)
    if use_cuda:
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(params, **kwargs)

