#!/usr/bin/env python3
"""Train and evaluate slice vs patch arms on synthetic fine-grained tasks.

Examples:
  python scripts/train_benchmark.py --task needle --steps 1500 --seeds 3
  python scripts/train_benchmark.py --task glyph --arms patch4,slice_loc_nogumbel
  python scripts/train_benchmark.py --bench --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.models import ARMS, build
from fine_grain.tasks import TASKS
from fine_grain.train_utils import (
    _build_train_pool,
    _make_optimizer,
    collapse_stats,
    load_kinks_folder,
    pr_obj,
)

def run_arm(name, spec, args, seed, task):
    torch.manual_seed(seed)
    dev = torch.device(args.device)
    make, n_cls = task["fn"], task["n_cls"]
    model = build(spec, args.dim, args.depth, args.slice_num, n_cls).to(dev)
    use_cuda = dev.type == "cuda"
    use_amp = bool(getattr(args, "amp", False) and use_cuda)
    use_compile = bool(getattr(args, "compile", False) and use_cuda)
    # Compile AFTER .to(cuda). reduce-overhead helps small-batch decode-like loops;
    # default is safer if reduce-overhead fails on Windows/Triton.
    if use_compile:
        mode = getattr(args, "compile_mode", "default") or "default"
        try:
            model = torch.compile(model, mode=mode)
        except Exception as e:
            print(f"  [warn] torch.compile({mode}) failed ({e}); running eager", flush=True)
            use_compile = False
    mult, sizes = spec["mult"], np.array(args.sizes)
    n_par_est = sum(p.numel() for p in model.parameters())
    # Budget rules (optional; 0 = use --steps only):
    #   data_mult  D:  samples_seen = steps*batch ≥ D * n_params
    #   token_mult T:  tokens_seen  = steps*batch*tok_per_img ≥ T * n_params
    #     tok_per_img = (res/patch)^2 for patch arms, else res^2 (point stream) for slice.
    # "T=10 tokens per param" is a light Chinchilla-style rule of thumb; OK for this toy.
    data_mult = float(getattr(args, "data_mult", 0) or 0)
    token_mult = float(getattr(args, "token_mult", 0) or 0)
    steps = int(args.steps)
    if spec["kind"] == "patch":
        tok_per = max(1, (args.res * mult // spec["patch"]) ** 2)
    else:
        tok_per = max(1, (args.res * mult) ** 2)
    if data_mult > 0:
        steps = max(steps, int(math.ceil(data_mult * n_par_est / max(1, args.batch))))
    if token_mult > 0:
        steps = max(steps, int(math.ceil(
            token_mult * n_par_est / max(1, args.batch * tok_per))))
    if data_mult > 0 or token_mult > 0:
        print(f"  [{name}] n_par={n_par_est} tok/img={tok_per} → steps={steps} "
              f"samples≈{steps * args.batch} tokens≈{steps * args.batch * tok_per}",
              flush=True)
    opt = _make_optimizer(model.parameters(), args.lr, use_cuda)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.default_rng(1000 + seed)                # train generator
    data_dir = getattr(args, "data_dir", "") or ""
    t0 = time.time()
    if data_dir:
        # on-disk materialised split (build_kinks_dataset.py)
        split = os.path.join(data_dir, "train")
        p_img, p_lab = load_kinks_folder(split)
        pool_n = int(p_img.shape[0])
        bytes_est = int(p_img.nelement() * p_img.element_size())
        pool_on_gpu = use_cuda and bytes_est <= int(getattr(args, "pool_gpu_max_mb", 512)) * 1024 ** 2
        if pool_on_gpu:
            p_img = p_img.to(dev, non_blocking=True)
            p_lab = p_lab.to(dev, non_blocking=True)
            if use_cuda:
                torch.cuda.synchronize()
        t_pool = time.time() - t0
        print(f"  [{name}] data_dir={split}  N={pool_n} on_{'gpu' if pool_on_gpu else 'cpu'} "
              f"(~{bytes_est/1024**2:.0f} MiB)", flush=True)
    else:
        # Pool: default scales with batch; at high res keep pool on CPU (256²×8k ≈ multi-GB).
        pool_n = int(getattr(args, "pool", 0) or max(4096, 32 * args.batch))
        pool_n = (pool_n // args.batch) * args.batch
        bytes_est = pool_n * 3 * (args.res * mult) ** 2 * 4
        pool_on_gpu = use_cuda and bytes_est <= int(getattr(args, "pool_gpu_max_mb", 512)) * 1024 ** 2
        p_img, p_lab = _build_train_pool(make, rng, sizes, args.res, mult, pool_n, args.batch)
        if pool_on_gpu:
            p_img = p_img.to(dev, non_blocking=True)
            p_lab = p_lab.to(dev, non_blocking=True)
            if use_cuda:
                torch.cuda.synchronize()
        t_pool = time.time() - t0
        print(f"  [{name}] pool_n={pool_n} on_{'gpu' if pool_on_gpu else 'cpu'} "
              f"(~{bytes_est/1024**2:.0f} MiB raw)", flush=True)

    def _batch_xy(idx_cpu_or_gpu):
        if pool_on_gpu:
            return p_img[idx_cpu_or_gpu], p_lab[idx_cpu_or_gpu]
        idx = idx_cpu_or_gpu.cpu()
        return p_img[idx].to(dev, non_blocking=use_cuda), p_lab[idx].to(dev, non_blocking=use_cuda)

    # Warmup: absorb cudnn autotune + compile first-graph cost so train timing is honest
    # and the first real step is not a multi-minute stall mid-run.
    model.train()
    t_warm = 0.0
    warm_n = int(getattr(args, "warmup", 0) or (8 if use_compile else (3 if use_cuda else 0)))
    if warm_n and use_cuda:
        tw = time.time()
        g_w = torch.Generator(device="cpu")
        g_w.manual_seed(0)
        for _ in range(warm_n):
            idx = torch.randint(0, pool_n, (args.batch,), generator=g_w)
            if pool_on_gpu:
                idx = idx.to(dev)
            xb, yb = _batch_xy(idx)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        t_warm = time.time() - tw

    t0 = time.time()
    g = torch.Generator(device="cpu")
    g.manual_seed(1000 + seed)
    for _ in range(steps):
        idx = torch.randint(0, pool_n, (args.batch,), generator=g)
        if pool_on_gpu:
            idx = idx.to(dev)
        xb, yb = _batch_xy(idx)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    if use_cuda:
        torch.cuda.synchronize()
    t_train = time.time() - t0

    model.eval()
    accs, prs = {}, []
    col = None
    # optional on-disk val split
    v_img = v_lab = None
    if data_dir:
        val_split = os.path.join(data_dir, "val")
        if os.path.isdir(val_split) and os.path.isfile(os.path.join(val_split, "labels.csv")):
            v_img, v_lab = load_kinks_folder(val_split)
            print(f"  [{name}] val from {val_split}  N={len(v_lab)}", flush=True)
    with torch.no_grad():
        for s in args.sizes:
            hit = tot = 0
            if v_img is not None:
                # class for kinks: s - 5 when k_min=5
                cls = int(s) - 5 if args.task == "kinks" else int(s)
                idx = (v_lab == cls).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    accs[s] = float("nan")
                    continue
                for st in range(0, len(idx), args.batch):
                    sel = idx[st:st + args.batch]
                    img = v_img[sel].to(dev, non_blocking=use_cuda)
                    lab = v_lab[sel].to(dev, non_blocking=use_cuda)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model(img).argmax(-1)
                    hit += int((pred == lab).sum()); tot += len(lab)
                    if spec["kind"] == "slice":
                        col = collapse_stats(model)
            else:
                ev = np.random.default_rng(90000 + s)       # held-out online generator
                for _ in range(max(1, args.eval_n // args.batch)):
                    img, lab, msk = make(ev, np.full(args.batch, s), args.res, mult)
                    img = img.to(dev, non_blocking=use_cuda)
                    lab = lab.to(dev, non_blocking=use_cuda)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model(img).argmax(-1)
                    hit += int((pred == lab).sum()); tot += len(lab)
                    if spec["kind"] == "slice":
                        prs.append(pr_obj(model, msk))
                        col = collapse_stats(model)
            accs[s] = hit / tot if tot else float("nan")
            if spec["kind"] == "slice" and v_img is not None and col is None:
                # one forward for collapse if never set
                pass
    n_tok = ((args.res * mult) ** 2 if spec.get("readout") == "points"
             else args.slice_num if spec["kind"] == "slice"
             else (args.res * mult // spec["patch"]) ** 2)
    pr = float(np.mean(prs)) if prs else float("nan")

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Under compile, parameter count may walk wrappers; fall back to raw build size.
    if n_par == 0:
        n_par = sum(p.numel() for p in build(spec, args.dim, args.depth,
                                             args.slice_num, n_cls).parameters())
    flags = []
    if use_compile:
        flags.append("compile")
    if use_amp:
        flags.append("amp")
    flag_s = ("[" + ",".join(flags) + "] ") if flags else ""
    print(f"  {name:<12} seed{seed}  {t_pool+t_warm+t_train:5.0f}s  "
          f"(pool {t_pool:.1f}s + warm {t_warm:.1f}s + train {t_train:.1f}s)  "
          f"{flag_s}{n_par/1e3:5.1f}k par  tok={n_tok:<4} PR_obj={pr:5.2f}  "
          + "  ".join(f"s{s}={accs[s]:.3f}" for s in args.sizes),
          flush=True)
    if col is not None:
        print(f"    collapse: PR_mass={col['PR_mass']:.2f}  H_mass={col['H_mass']:.3f}  "
              f"H_point={col['H_point']:.3f}  cos_tok={col['cos_tok']:.3f}  "
              f"r99={col['r99']:.1f}/{args.slice_num}",
              flush=True)
    # always save a checkpoint so probes (τ / collapse) can load without retrain
    ckpt_dir = getattr(args, "ckpt_dir", "") or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    # make_kinks kwargs are not in ARMS; record task + generator version tags in meta
    ckpt_path = os.path.join(
        ckpt_dir, f"{args.task}_{name}_seed{seed}_res{args.res}.pt")
    torch.save({
        "arm": name,
        "spec": spec,
        "task": args.task,
        "seed": seed,
        "res": args.res,
        "dim": args.dim,
        "depth": args.depth,
        "slice_num": args.slice_num,
        "steps": steps,
        "batch": args.batch,
        "accs": accs,
        "collapse": col,
        "state_dict": raw.state_dict(),
        "n_par": n_par,
    }, ckpt_path)
    print(f"    ckpt → {ckpt_path}", flush=True)
    return accs, n_tok, pr


def bench_speed(args):
    """Short wall-clock grid over batch / amp / compile. Run before long sweeps."""
    print("=== speed bench (slice_nogumbel, timed train only) ===", flush=True)
    dev = torch.device(args.device)
    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    configs = []
    for batch in args.bench_batches:
        for amp in ([False, True] if use_cuda else [False]):
            for compile_on in ([False, True] if use_cuda and args.bench_compile else [False]):
                configs.append((batch, amp, compile_on))
    # Fixed tiny pool; only measures steady-state step time after warmup.
    results = []
    for batch, amp, compile_on in configs:
        torch.cuda.empty_cache() if use_cuda else None
        model = build(ARMS["slice_nogumbel"], args.dim, args.depth,
                      args.slice_num, 4).to(dev)
        if compile_on:
            try:
                model = torch.compile(model, mode=args.compile_mode)
            except Exception as e:
                print(f"  batch={batch} amp={amp} compile=1 FAIL compile: {e}", flush=True)
                continue
        opt = _make_optimizer(model.parameters(), args.lr, use_cuda)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        img = torch.rand(batch, 3, args.res, args.res, device=dev)
        lab = torch.randint(0, 4, (batch,), device=dev)
        model.train()
        # warmup
        for _ in range(8 if compile_on else 3):
            with torch.amp.autocast("cuda", enabled=amp):
                loss = F.cross_entropy(model(img), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if use_cuda:
            torch.cuda.synchronize()
        n_step = args.bench_steps
        t0 = time.time()
        try:
            for _ in range(n_step):
                with torch.amp.autocast("cuda", enabled=amp):
                    loss = F.cross_entropy(model(img), lab)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            if use_cuda:
                torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"  batch={batch:<4} amp={int(amp)} compile={int(compile_on)}  OOM/FAIL {e}",
                  flush=True)
            del model, opt, img, lab
            torch.cuda.empty_cache() if use_cuda else None
            continue
        dt = time.time() - t0
        ms = 1000.0 * dt / n_step
        samp_s = batch * n_step / dt
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if use_cuda else 0
        line = (f"  batch={batch:<4} amp={int(amp)} compile={int(compile_on)}  "
                f"{ms:7.1f} ms/step  {samp_s:8.0f} samp/s  peak={peak:6.0f} MiB")
        print(line, flush=True)
        results.append((samp_s, batch, amp, compile_on, ms, peak))
        del model, opt, img, lab
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    if results:
        best = max(results, key=lambda r: r[0])
        print(f"\nBEST by samp/s: batch={best[1]} amp={int(best[2])} compile={int(best[3])}  "
              f"{best[4]:.1f} ms/step  {best[0]:.0f} samp/s  peak={best[5]:.0f} MiB",
              flush=True)
        print(f"Suggested: --batch {best[1]}"
              f"{' --amp' if best[2] else ''}"
              f"{' --compile' if best[3] else ''}"
              f" --pool 8192 --seeds 1", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="needle", choices=list(TASKS))
    ap.add_argument("--arms", default="patch4,patch8,patch4_hi,slice,slice_const,slice_sum")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="default = the task's own sweep (see TASKS)")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", type=int, default=0,
                    help="train image pool size (0 = auto max(4096, 32*batch))")
    ap.add_argument("--data_mult", type=float, default=0,
                    help="if >0, samples_seen=steps*batch ≥ data_mult * n_params")
    ap.add_argument("--token_mult", type=float, default=0,
                    help="if >0, tokens_seen=steps*batch*tok_per_img ≥ token_mult * n_params "
                         "(10 ≈ ten training tokens per parameter)")
    ap.add_argument("--pool_gpu_max_mb", type=int, default=512,
                    help="keep train pool on GPU only if raw pool bytes ≤ this many MiB")
    ap.add_argument("--ckpt_dir", default="checkpoints",
                    help="directory for per-arm .pt checkpoints (always saved after eval)")
    ap.add_argument("--data_dir", default="",
                    help="if set (e.g. data/kinks256), load train/ from build_kinks_dataset.py "
                         "instead of online synthetic pool")
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--slice_num", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="CUDA autocast fp16 + GradScaler")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile (Inductor/Triton when available)")
    ap.add_argument("--compile_mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode")
    ap.add_argument("--warmup", type=int, default=0,
                    help="extra train steps before timed loop (0=auto)")
    ap.add_argument("--bench", action="store_true",
                    help="speed grid only; do not run full arms")
    ap.add_argument("--bench_steps", type=int, default=40)
    ap.add_argument("--bench_batches", type=int, nargs="+", default=[64, 128, 160, 192])
    ap.add_argument("--bench_compile", action="store_true", default=True,
                    help="include compile in --bench (default on)")
    ap.add_argument("--no_bench_compile", action="store_true",
                    help="skip compile configs in --bench")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.no_bench_compile:
        args.bench_compile = False
    if args.bench:
        bench_speed(args)
        return
    task = TASKS[args.task]
    if args.sizes is None:
        args.sizes = task["sizes"]

    print(f"task={args.task} | res {args.res} | steps {args.steps} | batch {args.batch} | "
          f"chance = {1/task['n_cls']:.2f} | seeds {args.seeds} | device {args.device}"
          f" | amp={int(args.amp)} compile={int(args.compile)}",
          flush=True)
    print(flush=True)

    table = {}
    for name in args.arms.split(","):
        runs = [run_arm(name, ARMS[name], args, sd, task) for sd in range(args.seeds)]
        table[name] = ([r[0] for r in runs], runs[0][1],
                       float(np.mean([r[2] for r in runs])))

    print(f"\n{'arm':<12} {'tok':>4} {'PR_obj':>7} | "
          + " ".join(f"{'s='+str(s):>12}" for s in args.sizes), flush=True)
    for name, (accs, tok, pr) in table.items():
        cells = []
        for s in args.sizes:
            v = np.array([a[s] for a in accs])
            cells.append(f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1
                         else f"{v.mean():.3f}       ")
        print(f"{name:<12} {tok:>4} {pr:>7.2f} | " + " ".join(f"{c:>12}" for c in cells),
              flush=True)

    print("\nv1: P1 patch cliff tracks patch size | P2 slice flat to s=1 | "
          "P3 slice_sum degrades | P4 patch4_hi buys ~1 octave at 4x tokens\n"
          "v2: P5' PR_obj~1 => shape-blind, slice should lose on glyph; PR_obj>>1 => "
          "parts decomposition | P6 lines: patch arms fine, s/p law does NOT transfer | "
          "P7 slice_const lands strictly between slice and slice_sum | "
          "P8 connect: no arm clears 0.65", flush=True)


if __name__ == "__main__":
    main()
