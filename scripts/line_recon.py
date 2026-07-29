"""Line reconstruction: image -> polyline mask (BCE + Dice).

Default input is luminance×3 (blocks pure-red channel shortcut on RGB red lines).
Decoder is arm-faithful:
  - slice: Linear on final point stream (post deslice residual) -> HxW logits
  - patch: tokens on grid -> bilinear upsample to HxW -> 1x1 conv

Example:
  python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fine_grain as D


def to_gray3(img_rgb: torch.Tensor) -> torch.Tensor:
    """RGB [B,3,H,W] -> luminance stacked to 3 channels (kill pure-red shortcut)."""
    # Rec. 601 luma
    r, g, b = img_rgb[:, 0], img_rgb[:, 1], img_rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def make_batch(rng, B, res, hard_frac, hard_tile, k_list, rgb=False):
    """rgb=False: luminance×3 (legacy). rgb=True: raw RGB with pure-red polyline."""
    ks = rng.choice(k_list, size=B)
    imgs, masks = [], []
    for k in ks:
        im, _lab, msk = D.make_kinks(
            rng, np.array([int(k)]), res=res,
            hard_tile=hard_tile, hard_frac=hard_frac,
        )
        # make_kinks/_done returns mask as [1, H*W] tensor (bool/float)
        if torch.is_tensor(msk):
            m = msk.float().reshape(-1, res, res)
        else:
            m = torch.from_numpy(np.asarray(msk, np.float32)).reshape(-1, res, res)
        imgs.append(im)
        masks.append(m)
    img = torch.cat(imgs, 0)                 # [B,3,H,W]
    m = torch.cat(masks, 0)                  # [B,H,W]
    if rgb:
        return img, m
    return to_gray3(img), m


class SliceSeg(nn.Module):
    """Slice encoder + per-point 1x logit (uses point stream after blocks)."""

    def __init__(self, dim=64, depth=3, slice_num=32, local=True, nog=True):
        super().__init__()
        self.inner = D.SliceNet(
            dim=dim, depth=depth, slice_num=slice_num, norm="mass",
            n_cls=2, readout="points", local=local, n_freq=0,
        )
        if nog:
            for b in self.inner.blocks:
                b.mix.no_gumbel = True
        # replace cls head with per-point logit
        self.inner.pool = nn.Identity()
        self.inner.head = nn.Identity()
        self.pt_head = nn.Linear(dim, 1)

    def forward(self, img):
        B, _, R, _ = img.shape
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = D.coords(R, img.device)
        x = self.inner.stem(torch.cat([pts, p.expand(B, -1, -1)], -1))
        if self.inner.local is not None:
            g = x.transpose(1, 2).reshape(B, -1, R, R)
            x = x + self.inner.local(g).flatten(2).transpose(1, 2)
        for b in self.inner.blocks:
            x, _ = b(x)
        logits = self.pt_head(x).squeeze(-1).reshape(B, R, R)
        return logits


class PatchSeg(nn.Module):
    """Patch encoder + upsample tokens to full-res mask logits."""

    def __init__(self, dim=64, depth=3, patch=16):
        super().__init__()
        self.patch = patch
        self.inner = D.PatchNet(dim=dim, depth=depth, patch=patch, n_cls=2)
        self.inner.pool = nn.Identity()
        self.inner.head = nn.Identity()
        self.proj = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, img):
        B, _, R, _ = img.shape
        f = self.inner.stem(img)
        Rp = f.shape[-1]
        x = f.reshape(B, -1, Rp * Rp).transpose(1, 2)
        x = torch.cat([x, D.coords(Rp, img.device).expand(B, -1, -1)], -1)
        for b in self.inner.blocks:
            x, _ = b(x)
        # x: [B, T, dim] on Rp x Rp grid
        feat = x.transpose(1, 2).reshape(B, -1, Rp, Rp)
        up = F.interpolate(feat, size=(R, R), mode="bilinear", align_corners=False)
        return self.proj(up).squeeze(1)


def dice_loss_with_logits(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    t = target
    inter = (p * t).sum(dim=(1, 2))
    den = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def metrics(logits, target, thr=0.5):
    p = (logits.sigmoid() > thr).float()
    t = target
    # pixel acc
    acc = (p == t).float().mean().item()
    inter = (p * t).sum().item()
    union = ((p + t) > 0).float().sum().item()
    iou = inter / max(union, 1.0)
    dice = (2 * inter) / max(p.sum().item() + t.sum().item(), 1.0)
    # recall on line pixels
    pos = t.sum().item()
    rec = inter / max(pos, 1.0)
    prec = inter / max(p.sum().item(), 1.0)
    return dict(acc=acc, iou=iou, dice=dice, recall=rec, precision=prec)


def trivial_luma_baseline(img_gray3, target, thr=None):
    """Threshold luminance — upper bound on color/intensity shortcut (no learning)."""
    y = img_gray3[:, 0]
    if thr is None:
        # pick thr maximizing dice on this batch (oracle threshold — optimistic)
        best = dict(dice=0.0, thr=0.5)
        for t in np.linspace(0.05, 0.95, 19):
            logits = (y - t) * 20  # soft step
            m = metrics(logits, target)
            if m["dice"] > best["dice"]:
                best = {**m, "thr": float(t)}
        return best
    logits = (y - thr) * 20
    return metrics(logits, target)


def build_arm(name, dim, depth, slice_num):
    # Prefer ARMS-driven flags (topk deslice, Stiefel, gate, …) when name is registered.
    if name in D.ARMS and D.ARMS[name].get("kind") == "slice":
        spec = dict(D.ARMS[name])
        # SliceSeg needs points readout + local/nog from spec
        m = SliceSeg(
            dim=dim, depth=depth, slice_num=slice_num,
            local=bool(spec.get("local", False)),
            nog=bool(spec.get("nog", False)),
        )
        D.apply_slice_flags(m.inner, spec)
        return m
    if name in ("slice_loc_nogumbel", "slice_loc_nogumbel_st"):
        m = SliceSeg(dim=dim, depth=depth, slice_num=slice_num, local=True, nog=True)
        if name.endswith("_st") or "st" in name.split("_"):
            for b in m.inner.blocks:
                b.mix.stiefel_ns = True
                b.mix.ns_steps = D.NS_STEPS_DEFAULT
                b.mix.ns_coefficients = (D.NS_A, D.NS_B, D.NS_C)
                b.mix.ns_eps = D.NS_EPS
        return m
    if name == "slice_nogumbel":
        return SliceSeg(dim=dim, depth=depth, slice_num=slice_num, local=False, nog=True)
    if name.startswith("patch"):
        p = int(name.replace("patch", "") or "16")
        return PatchSeg(dim=dim, depth=depth, patch=p)
    raise ValueError(name)


def run_arm(name, args, device):
    torch.manual_seed(args.seed)
    model = build_arm(name, args.dim, args.depth, args.slice_num).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model)
            print(f"  [{name}] compile ok", flush=True)
        except Exception as e:
            print(f"  [{name}] compile fail: {e}", flush=True)

    opt = D._make_optimizer(model.parameters(), args.lr, device.type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # pos_weight ~ negatives/positives; 1px line on 64^2 ≈ 1/50–1/100
    pos_weight = torch.tensor([args.pos_weight], device=device)

    k_list = list(range(args.k_min, args.k_max + 1))
    rng = np.random.default_rng(1000 + args.seed)
    history = []
    t0 = time.time()
    print(
        f"  [{name}] n_par={n_par} B={args.batch} steps={args.steps} res={args.res} "
        f"hard_frac={args.hard_frac} pos_w={args.pos_weight}",
        flush=True,
    )

    def evaluate(tag_step):
        model.eval()
        rng_e = np.random.default_rng(90_000 + tag_step)
        dices, ious, recs = [], [], []
        base_dices = []
        n_left = args.eval_n
        while n_left > 0:
            b = min(args.eval_batch, n_left)
            x, m = make_batch(rng_e, b, args.res, args.hard_frac, args.hard_tile, k_list)
            x, m = x.to(device), m.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            met = metrics(logits.float(), m)
            dices.append(met["dice"])
            ious.append(met["iou"])
            recs.append(met["recall"])
            base_dices.append(trivial_luma_baseline(x.float(), m)["dice"])
            n_left -= b
        out = dict(
            step=tag_step,
            dice=float(np.mean(dices)),
            iou=float(np.mean(ious)),
            recall=float(np.mean(recs)),
            trivial_dice=float(np.mean(base_dices)),
            seconds=time.time() - t0,
        )
        model.train()
        return out

    # step-0 eval (init)
    st0 = evaluate(0)
    history.append(st0)
    print(
        f"  [{name}] step 0  dice={st0['dice']:.3f} iou={st0['iou']:.3f} "
        f"rec={st0['recall']:.3f}  trivial_dice={st0['trivial_dice']:.3f}",
        flush=True,
    )

    model.train()
    for step in range(1, args.steps + 1):
        x, m = make_batch(rng, args.batch, args.res, args.hard_frac, args.hard_tile, k_list)
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(
                logits, m, pos_weight=pos_weight,
            )
            dloss = dice_loss_with_logits(logits.float(), m)
            loss = bce + args.dice_w * dloss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.probe_every == 0 or step == args.steps:
            st = evaluate(step)
            st["loss"] = float(loss.item())
            st["bce"] = float(bce.item())
            history.append(st)
            print(
                f"  [{name}] step {step:4d}  loss={st['loss']:.3f}  "
                f"dice={st['dice']:.3f} iou={st['iou']:.3f} rec={st['recall']:.3f}  "
                f"trivial={st['trivial_dice']:.3f}",
                flush=True,
            )
            ckpt_dir = args.ckpt_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            raw = getattr(model, "_orig_mod", model)
            path = os.path.join(ckpt_dir, f"{name}_res{args.res}_seed{args.seed}_step{step:05d}.pt")
            torch.save({
                "arm": name, "step": step, "model": raw.state_dict(),
                "args": vars(args), "probe": st,
            }, path)

    return dict(arm=name, n_par=n_par, history=history, final=history[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="slice_loc_nogumbel,patch16,patch4")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--probe_every", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=256)
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
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ckpt_dir", default="checkpoints/line_recon_64")
    ap.add_argument("--out", default="results/line_recon_64.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"line-recon | res={args.res} B={args.batch} steps={args.steps} "
        f"arms={args.arms} hard_frac={args.hard_frac} amp={int(args.amp)} "
        f"compile={int(args.compile)} device={device}",
        flush=True,
    )
    print("input=luminance×3  target=polyline mask  loss=BCE(pos_w)+Dice", flush=True)

    results = []
    for name in args.arms.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        results.append(run_arm(name, args, device))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {
        "task": "line_recon_gray",
        "res": args.res,
        "batch": args.batch,
        "steps": args.steps,
        "hard_frac": args.hard_frac,
        "arms": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary (final dice) ===", flush=True)
    for r in results:
        f = r["final"]
        print(
            f"  {r['arm']:<22} dice={f['dice']:.3f}  iou={f['iou']:.3f}  "
            f"rec={f['recall']:.3f}  trivial={f['trivial_dice']:.3f}  par={r['n_par']}",
            flush=True,
        )
    print(f"json → {args.out}", flush=True)
    print(f"ckpt → {args.ckpt_dir}/", flush=True)


if __name__ == "__main__":
    main()
