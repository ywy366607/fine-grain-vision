"""Accelerated kernels for persistent Slice read/write.

The Triton path fuses point-to-Slice projection and row softmax. Reductions and
the analytic backward use PyTorch GEMMs so correctness remains easy to audit.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installations
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _assignment_softmax_kernel(
        points_ptr,
        weight_ptr,
        output_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        slices: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        BLOCK_SLICES: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        slice_offsets = tl.arange(0, BLOCK_SLICES)
        accumulator = tl.zeros((BLOCK_ROWS, BLOCK_SLICES), dtype=tl.float32)

        for channel_start in range(0, channels, BLOCK_CHANNELS):
            channel_offsets = channel_start + tl.arange(0, BLOCK_CHANNELS)
            points = tl.load(
                points_ptr
                + row_offsets[:, None] * channels
                + channel_offsets[None, :],
                mask=(row_offsets[:, None] < rows)
                & (channel_offsets[None, :] < channels),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + slice_offsets[:, None] * channels
                + channel_offsets[None, :],
                mask=(slice_offsets[:, None] < slices)
                & (channel_offsets[None, :] < channels),
                other=0.0,
            )
            # Assignment logits are sensitive to small errors before softmax.
            # Match PyTorch's full-precision FP32 reference instead of Triton's
            # default TF32 input precision on NVIDIA GPUs.
            accumulator += tl.dot(points, tl.trans(weight), input_precision="ieee")

        logits = accumulator * scale
        logits = tl.where(slice_offsets[None, :] < slices, logits, -float("inf"))
        logits = logits - tl.max(logits, axis=1)[:, None]
        numerator = tl.exp(logits)
        probability = numerator / tl.sum(numerator, axis=1)[:, None]
        tl.store(
            output_ptr + row_offsets[:, None] * slices + slice_offsets[None, :],
            probability,
            mask=(row_offsets[:, None] < rows)
            & (slice_offsets[None, :] < slices),
        )


class _TritonAssignmentSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, points: torch.Tensor, weight: torch.Tensor, scale: float
    ) -> torch.Tensor:
        if triton is None or not points.is_cuda:
            raise RuntimeError("Triton assignment requires a CUDA Triton installation")
        if points.shape[-1] != weight.shape[-1]:
            raise ValueError("point and assignment channel widths must match")
        original_shape = points.shape[:-1]
        flat_points = points.contiguous().reshape(-1, points.shape[-1])
        weight = weight.contiguous()
        output = torch.empty(
            flat_points.shape[0],
            weight.shape[0],
            device=points.device,
            dtype=points.dtype,
        )
        block_slices = triton.next_power_of_2(weight.shape[0])
        if block_slices > 512:
            raise ValueError("Triton assignment currently supports at most 512 slices")
        block_channels = min(64, triton.next_power_of_2(weight.shape[1]))
        block_channels = max(16, block_channels)
        # Keep the logits tile below the shared-memory limit on consumer GPUs.
        # Large Slice banks retain full capacity; only row-level parallelism is
        # reduced, and the launch grid exposes the remaining parallel work.
        block_rows = 8 if block_slices >= 256 else 16
        grid = (triton.cdiv(flat_points.shape[0], block_rows),)
        _assignment_softmax_kernel[grid](
            flat_points,
            weight,
            output,
            rows=flat_points.shape[0],
            channels=flat_points.shape[1],
            slices=weight.shape[0],
            scale=float(scale),
            BLOCK_ROWS=block_rows,
            BLOCK_CHANNELS=block_channels,
            BLOCK_SLICES=block_slices,
            num_warps=8 if block_slices >= 256 else 4,
            num_stages=1,
        )
        ctx.save_for_backward(flat_points, weight, output)
        ctx.scale = float(scale)
        ctx.original_shape = original_shape
        return output.reshape(*original_shape, weight.shape[0])

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        points, weight, probability = ctx.saved_tensors
        gradient = gradient.contiguous().reshape_as(probability)
        dot = (gradient * probability).sum(dim=-1, keepdim=True)
        logits_gradient = probability * (gradient - dot)
        point_gradient = (logits_gradient @ weight) * ctx.scale
        weight_gradient = (logits_gradient.transpose(0, 1) @ points) * ctx.scale
        return point_gradient.reshape(*ctx.original_shape, weight.shape[1]), weight_gradient, None


def slice_assignment(
    points: torch.Tensor,
    weight: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute data-dependent point-to-Slice probabilities.

    `auto` uses Triton for supported CUDA tensors and the torch path elsewhere.
    Both paths implement exactly `softmax(points @ weight.T / sqrt(C))`.
    """
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unknown Slice assignment backend: {backend}")
    use_triton = backend == "triton" or (
        backend == "auto"
        and triton is not None
        and points.is_cuda
        and points.dtype in {torch.float16, torch.bfloat16}
        and points.shape[-1] <= 128
        and points.shape[-1] % 16 == 0
        and weight.shape[0] <= 512
    )
    scale = 1.0 / math.sqrt(points.shape[-1])
    if use_triton:
        return _TritonAssignmentSoftmax.apply(points, weight, scale)
    if backend == "triton":
        raise RuntimeError("the requested Triton backend is unavailable for this tensor")
    return F.softmax(F.linear(points, weight) * scale, dim=-1)
