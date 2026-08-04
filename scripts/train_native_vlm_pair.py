#!/usr/bin/env python3
"""Train one arm of the deterministic Native-Slice versus Patch-MoT pair."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.native_vlm import SliceMoTConfig, SliceMoTVLM  # noqa: E402
from fine_grain.paired_init import initialize_paired_common  # noqa: E402
from fine_grain.patch_vlm import PatchMoTConfig, PatchMoTVLM  # noqa: E402
from fine_grain.pretrain_data import (  # noqa: E402
    ConsecutiveModalityBatchSampler,
    PretrainBatch,
    PretrainCollator,
    SQLiteVLMData,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slice_ablation_arguments(name: str) -> dict[str, object]:
    arguments = {
        "standardize_thin_detail": False,
        "point_adaptive_temperature": False,
        "gumbel_assignment": False,
        "stiefel_slices": False,
        "deslice_topk": 0,
    }
    if name == "detailstd":
        arguments["standardize_thin_detail"] = True
    elif name == "point_temp":
        arguments["point_adaptive_temperature"] = True
    elif name == "gumbel":
        arguments["gumbel_assignment"] = True
    elif name == "stiefel_ns":
        arguments["stiefel_slices"] = True
    elif name == "topk2_write":
        arguments["deslice_topk"] = 2
    elif name == "stiefel_point_temp":
        arguments["stiefel_slices"] = True
        arguments["point_adaptive_temperature"] = True
    elif name != "base":
        raise ValueError(f"unknown Slice ablation: {name}")
    return arguments


def build_model(arm: str, *, image_size: int, tiny: bool, slice_ablation: str):
    slice_arguments = slice_ablation_arguments(slice_ablation)
    if tiny:
        if arm == "slice":
            config = SliceMoTConfig(
                layers=2,
                point_width=32,
                model_width=64,
                visual_slices=16,
                attention_heads=4,
                visual_ffn_width=128,
                text_ffn_width=128,
                tile_points=1024,
                **slice_arguments,
            )
            return SliceMoTVLM(config)
        config = PatchMoTConfig(
            layers=2,
            model_width=64,
            visual_tokens=16,
            attention_heads=4,
            visual_ffn_width=128,
            text_ffn_width=128,
            patch_ffn_width=64,
            image_size=image_size,
            patch_size=image_size // 4,
        )
        return PatchMoTVLM(config)
    if arm == "slice":
        return SliceMoTVLM(SliceMoTConfig(**slice_arguments))
    return PatchMoTVLM(
        PatchMoTConfig(image_size=image_size, patch_size=image_size // 16)
    )


def make_optimizer(model, *, learning_rate: float, weight_decay: float, cuda: bool):
    arguments = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": weight_decay,
    }
    if cuda:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **arguments)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(model.parameters(), **arguments)


def learning_rate_at(
    consumed_tokens: int,
    *,
    total_tokens: int,
    peak: float,
    warmup_fraction: float,
) -> float:
    warmup = max(1, int(total_tokens * warmup_fraction))
    if consumed_tokens < warmup:
        return peak * consumed_tokens / warmup
    progress = min(1.0, (consumed_tokens - warmup) / max(1, total_tokens - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def move_batch(batch: PretrainBatch, device: torch.device) -> PretrainBatch:
    token_values = {
        name: getattr(batch.tokens, name).to(device, non_blocking=True)
        for name in (
            "prompt_ids",
            "prompt_mask",
            "target_input_ids",
            "target_labels",
            "target_mask",
        )
    }
    images = None
    if batch.images is not None:
        images = batch.images.to(device, non_blocking=True)
    return PretrainBatch(
        tokens=type(batch.tokens)(**token_values),
        images=images,
        sample_ids=batch.sample_ids,
        sources=batch.sources,
        tasks=batch.tasks,
        target_tokens=batch.target_tokens,
    )


def divide_gradients(model, denominator: int) -> None:
    inverse = 1.0 / denominator
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)


def atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("slice", "patch"), required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--tokens-per-update", type=int, default=32768)
    parser.add_argument("--total-target-tokens", type=int, default=64_000_000)
    parser.add_argument("--peak-learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--log-every-updates", type=int, default=10)
    parser.add_argument("--save-every-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--slice-ablation",
        choices=(
            "base",
            "detailstd",
            "point_temp",
            "gumbel",
            "stiefel_ns",
            "topk2_write",
            "stiefel_point_temp",
        ),
        default="stiefel_point_temp",
    )
    args = parser.parse_args()

    if not 0.0 <= args.warmup_fraction < 1.0:
        parser.error("--warmup-fraction must be in [0, 1)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_hash = file_sha256(args.schedule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    model = build_model(
        args.arm,
        image_size=args.image_size,
        tiny=args.tiny,
        slice_ablation=args.slice_ablation,
    )
    initialize_paired_common(model, args.seed)
    model.to(device)
    optimizer = make_optimizer(
        model,
        learning_rate=args.peak_learning_rate,
        weight_decay=args.weight_decay,
        cuda=device.type == "cuda",
    )
    progress = {
        "next_sample_index": 0,
        "target_tokens": 0,
        "optimizer_updates": 0,
    }
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        checkpoint_data = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint_data["arm"] != args.arm:
            raise ValueError("checkpoint arm does not match --arm")
        if checkpoint_data["schedule_sha256"] != schedule_hash:
            raise ValueError("checkpoint schedule does not match --schedule")
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        progress.update(checkpoint_data["progress"])
        torch.set_rng_state(checkpoint_data["torch_rng_state"])
        if device.type == "cuda" and checkpoint_data.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint_data["cuda_rng_state"])

    uncompiled_model = model
    if args.compile:
        model = torch.compile(model, dynamic=True)
    dataset = SQLiteVLMData(args.database, args.schedule)
    sampler = ConsecutiveModalityBatchSampler(
        dataset.schedule,
        batch_size=args.micro_batch_size,
        start_index=int(progress["next_sample_index"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=PretrainCollator(image_size=args.image_size),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    metadata = {
        "arm": args.arm,
        "arguments": vars(args),
        "model_config": asdict(uncompiled_model.config),
        "parameters": uncompiled_model.parameter_count(),
        "schedule_sha256": schedule_hash,
        "schedule_samples": len(dataset),
        "device": str(device),
        "torch": torch.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def save_checkpoint(name: str) -> None:
        payload = {
            **metadata,
            "model": uncompiled_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "progress": dict(progress),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        }
        atomic_checkpoint(output_dir / name, payload)

    def amp_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_tokens = 0
    accumulated_loss = 0.0
    samples_this_run = 0
    next_save = (
        (int(progress["target_tokens"]) // args.save_every_tokens + 1)
        * args.save_every_tokens
    )
    started = time.monotonic()

    for batch in loader:
        if int(progress["target_tokens"]) >= args.total_target_tokens:
            break
        if args.max_samples is not None and samples_this_run >= args.max_samples:
            break
        batch = move_batch(batch, device)
        with amp_context():
            output = model(
                prompt_ids=batch.tokens.prompt_ids,
                target_input_ids=batch.tokens.target_input_ids,
                prompt_mask=batch.tokens.prompt_mask,
                target_mask=batch.tokens.target_mask,
                images=batch.images,
                labels=batch.tokens.target_labels,
                collect_diagnostics=False,
            )
            summed_loss = output.loss * batch.target_tokens
        summed_loss.backward()
        accumulated_tokens += batch.target_tokens
        accumulated_loss += float(summed_loss.detach())
        count = len(batch.sample_ids)
        samples_this_run += count
        progress["next_sample_index"] = int(progress["next_sample_index"]) + count
        progress["target_tokens"] = int(progress["target_tokens"]) + batch.target_tokens

        final_sample = int(progress["next_sample_index"]) >= len(dataset)
        reached_limit = (
            args.max_samples is not None and samples_this_run >= args.max_samples
        )
        if accumulated_tokens < args.tokens_per_update and not final_sample and not reached_limit:
            continue

        divide_gradients(uncompiled_model, accumulated_tokens)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            uncompiled_model.parameters(), args.gradient_clip
        )
        lr = learning_rate_at(
            int(progress["target_tokens"]),
            total_tokens=args.total_target_tokens,
            peak=args.peak_learning_rate,
            warmup_fraction=args.warmup_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress["optimizer_updates"] = int(progress["optimizer_updates"]) + 1
        elapsed = time.monotonic() - started
        row = {
            **progress,
            "loss": accumulated_loss / accumulated_tokens,
            "learning_rate": lr,
            "gradient_norm": float(gradient_norm),
            "tokens_per_second": (
                int(progress["target_tokens"]) / max(elapsed, 1e-6)
            ),
            "elapsed_seconds": elapsed,
        }
        append_jsonl(output_dir / "train.jsonl", row)
        if int(progress["optimizer_updates"]) % args.log_every_updates == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        accumulated_tokens = 0
        accumulated_loss = 0.0

        if int(progress["target_tokens"]) >= next_save:
            save_checkpoint("latest.pt")
            next_save += args.save_every_tokens

    if accumulated_tokens:
        divide_gradients(uncompiled_model, accumulated_tokens)
        torch.nn.utils.clip_grad_norm_(uncompiled_model.parameters(), args.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress["optimizer_updates"] = int(progress["optimizer_updates"]) + 1
    save_checkpoint("final.pt")
    print(json.dumps({"status": "complete", **progress}), flush=True)


if __name__ == "__main__":
    main()
