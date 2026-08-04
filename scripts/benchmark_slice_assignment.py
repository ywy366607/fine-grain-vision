#!/usr/bin/env python3
"""Benchmark the production-shape Slice assignment backends on CUDA."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.slice_kernels import slice_assignment


def elapsed_ms(operation, iterations: int) -> float:
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--point-width", type=int, default=256)
    parser.add_argument("--slices", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = getattr(torch, args.dtype)
    points = torch.randn(
        args.batch_size,
        args.points,
        args.point_width,
        device="cuda",
        dtype=dtype,
    )
    weight = torch.randn(
        args.slices, args.point_width, device="cuda", dtype=dtype
    )
    print(
        f"shape={tuple(points.shape)}x{args.slices} dtype={args.dtype} "
        f"gpu={torch.cuda.get_device_name()}"
    )
    for backend in ("torch", "triton"):
        milliseconds = elapsed_ms(
            lambda: slice_assignment(points, weight, backend=backend),
            args.iterations,
        )
        print(f"{backend:7s} {milliseconds:.4f} ms")


if __name__ == "__main__":
    main()
