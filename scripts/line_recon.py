"""Line reconstruction: image -> polyline mask (BCE + Dice).

Default input is luminance×3 (blocks pure-red channel shortcut on RGB red lines).
Use --rgb for pure-red RGB input (main published protocol).

Decoder is arm-faithful:
  - slice: Linear on final point stream (post deslice residual) -> HxW logits
  - patch (default): token -> Linear(dim, p*p) -> unpatchify
    so the head can predict within-patch structure (fair vs bilinear upsample).
  - patch --patch-decoder bilinear: legacy bilinear + 1x1 (confounds encoder vs head).

Example:
  python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600 --rgb
  python scripts/line_recon.py --arms patch16 --patch-decoder bilinear  # legacy unfair head
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
    r, g, b = img_rgb[:, 0], img_rgb[:, 1], img_rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def make_batch(rng, B, res, hard_frac, hard_tile, k_list, rgb=False):
    """rgb=False: luminance×3. rgb=True: raw RGB with pure-red polyline."""
    ks = rng.choice(k_list, size=B)
    imgs, masks = [], []
    for k in ks:
        im, _lab, msk = D.make_kinks(
            rng, np.array([int(k)]), res=res,
            hard_tile=hard_tile, hard_frac=hard_frac,
        )
        if torch.is_tensor(msk):
            m = msk.float().reshape(-1, res, res)
        else:
            m = torch.from_numpy(np.asarray(msk, np.float32)).reshape(-1, res, res)
        imgs.append(im)
        masks.append(m)
    img = torch.cat(imgs, 0)
    m = torch.cat(masks, 0)
    if rgb:
        return img, m
    return to_gray3(img), m


class SliceSeg(nn.Module):
    """Slice encoder + per-point logit (uses point stream after blocks)."""

    def __init__(self, dim=64, depth=3, slice_num=32, local=True, nog=True):
        super().__init__()
        self.inner = D.SliceNet(
            dim=dim, depth=depth, slice_num=slice_num, norm="mass",
            n_cls=2, readout="points", local=local, n_freq=0,
        )
        if nog:
            for b in self.inner.blocks:
                b.mix.no_gumbel = True
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
        return self.pt_head(x).squeeze(-1).reshape(B, R, R)


def unpatchify(pix: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, Hp, Wp, p*p] or [B, T, p*p] with square grid -> [B, H, W] logits.

    Layout matches ViT unpatchify: each token expands to a p×p spatial cell.
    """
    if pix.dim() == 3:
        B, T, pp = pix.shape
        side = int(T ** 0.5)
        assert side * side == T, f"token count {T} is not a square grid"
        assert pp == patch * patch, f"got {pp} channels, expected {patch*patch}"
        pix = pix.reshape(B, side, side, patch * patch)
    B, Hp, Wp, pp = pix.shape
    assert pp == patch * patch
    # [B, Hp, Wp, p, p] -> [B, Hp*p, Wp*p]
    pix = pix.reshape(B, Hp, Wp, patch, patch)
    pix = pix.permute(0, 1, 3, 2, 4).contiguous()
    return pix.reshape(B, Hp * patch, Wp * patch)


class PatchSeg(nn.Module):
    """Patch encoder + dense mask head.

    ``decoder`` modes
    -----------------
    unpatchify (default, fair)
        ``token -> Linear(dim, p*p) -> unpatchify``. Can express within-patch
        structure if the token carries it (Linear is a full-rank map when
        dim >= p*p; at dim=64, p=16 this is undercomplete — still far better
        than bilinear which cannot invent p×p modes at all).
    pixel_shuffle
        Same capacity as unpatchify via ``Conv2d + PixelShuffle(p)``.
    bilinear (legacy, unfair)
        ``tokens -> bilinear upsample -> 1x1 conv``. Smooths to patch-scale
        blobs; confounds encoder loss with head incapacity.
    """

    def __init__(self, dim=64, depth=3, patch=16, decoder="unpatchify"):
        super().__init__()
        assert decoder in ("unpatchify", "pixel_shuffle", "bilinear")
        self.patch = patch
        self.decoder = decoder
        self.inner = D.PatchNet(dim=dim, depth=depth, patch=patch, n_cls=2)
        self.inner.pool = nn.Identity()
        self.inner.head = nn.Identity()
        if decoder == "unpatchify":
            self.to_pixels = nn.Linear(dim, patch * patch)
        elif decoder == "pixel_shuffle":
            # r=patch: out channels must be r^2 for 1-channel mask
            self.to_grid = nn.Conv2d(dim, patch * patch, kernel_size=1)
            self.shuffle = nn.PixelShuffle(patch)
        else:
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
        if self.decoder == "bilinear":
            feat = x.transpose(1, 2).reshape(B, -1, Rp, Rp)
            up = F.interpolate(feat, size=(R, R), mode="bilinear", align_corners=False)
            logits = self.proj(up).squeeze(1)
        elif self.decoder == "pixel_shuffle":
            feat = x.transpose(1, 2).reshape(B, -1, Rp, Rp)
            logits = self.shuffle(self.to_grid(feat)).squeeze(1)
        else:
            logits = unpatchify(self.to_pixels(x), self.patch)
        if logits.shape[-2:] != (R, R):
            logits = F.interpolate(
                logits.unsqueeze(1), size=(R, R), mode="bilinear", align_corners=False,
            ).squeeze(1)
        return logits


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


def build_arm(name, dim, depth, slice_num, patch_decoder="unpatchify"):
    # Prefer ARMS-driven flags (topk deslice, Stiefel, gate, …) when name is registered.
    if name in D.ARMS and D.ARMS[name].get("kind") == "slice":
        spec = dict(D.ARMS[name])
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
        return PatchSeg(dim=dim, depth=depth, patch=p, decoder=patch_decoder)
    raise ValueError(name)


def run_arm(name, args, device):
    torch.manual_seed(args.seed)
    model = build_arm(
        name, args.dim, args.depth, args.slice_num,
        patch_decoder=args.patch_decoder,
    ).to(device)
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
    pos_weight = torch.tensor([args.pos_weight], device=device)

    k_list = list(range(args.k_min, args.k_max + 1))
    rng = np.random.default_rng(1000 + args.seed)
    history = []
    t0 = time.time()
    dec = args.patch_decoder if name.startswith("patch") else "point"
    print(
        f"  [{name}] n_par={n_par} decoder={dec} B={args.batch} steps={args.steps} "
        f"res={args.res} hard_frac={args.hard_frac} pos_w={args.pos_weight} "
        f"rgb={int(args.rgb)}",
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
            x, m = make_batch(
                rng_e, b, args.res, args.hard_frac, args.hard_tile, k_list, rgb=args.rgb,
            )
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

    st0 = evaluate(0)
    history.append(st0)
    print(
        f"  [{name}] step 0  dice={st0['dice']:.3f} iou={st0['iou']:.3f} "
        f"rec={st0['recall']:.3f}  trivial_dice={st0['trivial_dice']:.3f}",
        flush=True,
    )

    model.train()
    for step in range(1, args.steps + 1):
        x, m = make_batch(
            rng, args.batch, args.res, args.hard_frac, args.hard_tile, k_list, rgb=args.rgb,
        )
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
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument(
        "--rgb", action="store_true",
        help="raw RGB (pure-red polyline) instead of luminance×3",
    )
    ap.add_argument(
        "--patch-decoder", dest="patch_decoder", default="unpatchify",
        choices=["unpatchify", "pixel_shuffle", "bilinear"],
        help="patch mask head (default unpatchify is the fair head; "
             "bilinear is the legacy unfair baseline)",
    )
    ap.add_argument("--ckpt_dir", default="checkpoints/line_recon_64")
    ap.add_argument("--out", default="results/line_recon_64.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    mode = "rgb" if args.rgb else "luminance×3"
    print(
        f"line-recon | res={args.res} B={args.batch} steps={args.steps} "
        f"arms={args.arms} hard_frac={args.hard_frac} input={mode} "
        f"patch_decoder={args.patch_decoder} amp={int(args.amp)} "
        f"compile={int(args.compile)} device={device}",
        flush=True,
    )
    print("target=polyline mask  loss=BCE(pos_w)+Dice", flush=True)
    if args.patch_decoder == "bilinear":
        print(
            "WARN: --patch-decoder bilinear confounds encoder capacity with a "
            "head that cannot express within-patch structure.",
            flush=True,
        )

    results = []
    for name in args.arms.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        results.append(run_arm(name, args, device))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = {
        "task": "line_recon_rgb" if args.rgb else "line_recon_gray",
        "res": args.res,
        "batch": args.batch,
        "steps": args.steps,
        "hard_frac": args.hard_frac,
        "rgb": bool(args.rgb),
        "patch_decoder": args.patch_decoder,
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
