#!/usr/bin/env python3
"""Evaluate paired VLM checkpoints with matched, shuffled, and blank images."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.native_vlm import SliceMoTConfig, SliceMoTVLM  # noqa: E402
from fine_grain.patch_vlm import PatchMoTConfig, PatchMoTVLM  # noqa: E402
from fine_grain.pretrain_data import (  # noqa: E402
    ConsecutiveModalityBatchSampler,
    PretrainBatch,
    PretrainCollator,
    SQLiteVLMData,
)


def config_arguments(config_type, values: dict[str, object]) -> dict[str, object]:
    names = {item.name for item in fields(config_type)}
    return {key: value for key, value in values.items() if key in names}


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["arm"] == "slice":
        config = SliceMoTConfig(
            **config_arguments(SliceMoTConfig, checkpoint["model_config"])
        )
        model = SliceMoTVLM(config)
    else:
        config = PatchMoTConfig(
            **config_arguments(PatchMoTConfig, checkpoint["model_config"])
        )
        model = PatchMoTVLM(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def move_batch(batch: PretrainBatch, device: torch.device) -> PretrainBatch:
    tokens = type(batch.tokens)(
        **{
            name: getattr(batch.tokens, name).to(device, non_blocking=True)
            for name in (
                "prompt_ids",
                "prompt_mask",
                "target_input_ids",
                "target_labels",
                "target_mask",
            )
        }
    )
    return PretrainBatch(
        tokens=tokens,
        images=None if batch.images is None else batch.images.to(device, non_blocking=True),
        sample_ids=batch.sample_ids,
        sources=batch.sources,
        tasks=batch.tasks,
        target_tokens=batch.target_tokens,
    )


def per_sample_metrics(logits: torch.Tensor, labels: torch.Tensor):
    valid = labels != -100
    losses = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    token_count = valid.sum(dim=1)
    loss_sum = (losses * valid).sum(dim=1)
    correct = ((logits.argmax(dim=-1) == labels) & valid).sum(dim=1)
    first_index = valid.float().argmax(dim=1)
    batch_index = torch.arange(labels.shape[0], device=labels.device)
    first_loss = losses[batch_index, first_index]
    first_correct = (
        logits[batch_index, first_index].argmax(dim=-1)
        == labels[batch_index, first_index]
    )
    return loss_sum, correct, token_count, first_loss, first_correct


def add_metric(
    metrics: dict[str, dict[str, float]],
    source: str,
    name: str,
    value: float,
) -> None:
    row = metrics.setdefault(source, {})
    row[name] = row.get(name, 0.0) + value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument("--max-perturb-visual-samples", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = SQLiteVLMData(args.database, args.schedule)
    sampler = ConsecutiveModalityBatchSampler(
        dataset.schedule, batch_size=args.batch_size
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=PretrainCollator(image_size=args.image_size),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    metrics: dict[str, dict[str, float]] = {}
    evaluated_tokens = 0
    perturbed_visual_samples = 0

    def amp_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    with torch.inference_mode():
        for batch in loader:
            if args.max_target_tokens and evaluated_tokens >= args.max_target_tokens:
                break
            batch = move_batch(batch, device)
            with amp_context():
                matched = model(
                    prompt_ids=batch.tokens.prompt_ids,
                    target_input_ids=batch.tokens.target_input_ids,
                    prompt_mask=batch.tokens.prompt_mask,
                    target_mask=batch.tokens.target_mask,
                    images=batch.images,
                    collect_diagnostics=False,
                )
            loss_sum, correct, counts, first_loss, first_correct = per_sample_metrics(
                matched.logits.float(), batch.tokens.target_labels
            )
            for index, source in enumerate(batch.sources):
                count = int(counts[index])
                add_metric(metrics, source, "matched_loss_sum", float(loss_sum[index]))
                add_metric(metrics, source, "matched_correct", int(correct[index]))
                add_metric(metrics, source, "matched_first_loss_sum", float(first_loss[index]))
                add_metric(metrics, source, "matched_first_correct", int(first_correct[index]))
                add_metric(metrics, source, "tokens", count)
                add_metric(metrics, source, "samples", 1)
                evaluated_tokens += count

            can_perturb = (
                batch.images is not None
                and perturbed_visual_samples < args.max_perturb_visual_samples
            )
            if not can_perturb:
                continue
            blank_images = torch.ones_like(batch.images)
            with amp_context():
                blank = model(
                    prompt_ids=batch.tokens.prompt_ids,
                    target_input_ids=batch.tokens.target_input_ids,
                    prompt_mask=batch.tokens.prompt_mask,
                    target_mask=batch.tokens.target_mask,
                    images=blank_images,
                    collect_diagnostics=False,
                )
            blank_loss, _, _, blank_first_loss, _ = per_sample_metrics(
                blank.logits.float(), batch.tokens.target_labels
            )
            for index, source in enumerate(batch.sources):
                add_metric(metrics, source, "blank_loss_sum", float(blank_loss[index]))
                add_metric(metrics, source, "blank_tokens", int(counts[index]))
                add_metric(metrics, source, "blank_samples", 1)
                add_metric(
                    metrics, source, "blank_first_loss_sum", float(blank_first_loss[index])
                )

            if batch.images.shape[0] > 1:
                shuffled_images = batch.images.roll(1, dims=0)
                with amp_context():
                    shuffled = model(
                        prompt_ids=batch.tokens.prompt_ids,
                        target_input_ids=batch.tokens.target_input_ids,
                        prompt_mask=batch.tokens.prompt_mask,
                        target_mask=batch.tokens.target_mask,
                        images=shuffled_images,
                        collect_diagnostics=False,
                    )
                shuffled_loss, _, _, shuffled_first_loss, _ = per_sample_metrics(
                    shuffled.logits.float(), batch.tokens.target_labels
                )
                for index, source in enumerate(batch.sources):
                    add_metric(
                        metrics, source, "shuffled_loss_sum", float(shuffled_loss[index])
                    )
                    add_metric(metrics, source, "shuffled_tokens", int(counts[index]))
                    add_metric(metrics, source, "shuffled_samples", 1)
                    add_metric(
                        metrics,
                        source,
                        "shuffled_first_loss_sum",
                        float(shuffled_first_loss[index]),
                    )
            perturbed_visual_samples += batch.images.shape[0]

    total = {}
    for source, row in metrics.items():
        row["matched_nll"] = row["matched_loss_sum"] / row["tokens"]
        row["teacher_forced_byte_accuracy"] = row["matched_correct"] / row["tokens"]
        row["matched_first_byte_nll"] = row["matched_first_loss_sum"] / row["samples"]
        row["matched_first_byte_accuracy"] = row["matched_first_correct"] / row["samples"]
        if row.get("blank_tokens"):
            row["blank_nll"] = row["blank_loss_sum"] / row["blank_tokens"]
            row["blank_minus_matched_nll"] = row["blank_nll"] - row["matched_nll"]
            row["blank_first_byte_nll"] = (
                row["blank_first_loss_sum"] / row["blank_samples"]
            )
        if row.get("shuffled_tokens"):
            row["shuffled_nll"] = row["shuffled_loss_sum"] / row["shuffled_tokens"]
            row["shuffled_minus_matched_nll"] = (
                row["shuffled_nll"] - row["matched_nll"]
            )
            row["shuffled_first_byte_nll"] = (
                row["shuffled_first_loss_sum"] / row["shuffled_samples"]
            )
        for key, value in row.items():
            if key.endswith("_sum") or key in {
                "tokens",
                "matched_correct",
                "matched_first_correct",
                "samples",
                "blank_tokens",
                "blank_samples",
                "shuffled_tokens",
                "shuffled_samples",
            }:
                total[key] = total.get(key, 0.0) + value
    total["matched_nll"] = total["matched_loss_sum"] / total["tokens"]
    total["teacher_forced_byte_accuracy"] = total["matched_correct"] / total["tokens"]
    total["matched_first_byte_nll"] = total["matched_first_loss_sum"] / total["samples"]
    total["matched_first_byte_accuracy"] = (
        total["matched_first_correct"] / total["samples"]
    )
    if total.get("blank_tokens"):
        total["blank_nll"] = total["blank_loss_sum"] / total["blank_tokens"]
        total["blank_first_byte_nll"] = (
            total["blank_first_loss_sum"] / total["blank_samples"]
        )
    if total.get("shuffled_tokens"):
        total["shuffled_nll"] = total["shuffled_loss_sum"] / total["shuffled_tokens"]
        total["shuffled_first_byte_nll"] = (
            total["shuffled_first_loss_sum"] / total["shuffled_samples"]
        )
    report = {
        "arm": checkpoint["arm"],
        "checkpoint": args.checkpoint,
        "checkpoint_progress": checkpoint["progress"],
        "evaluated_tokens": evaluated_tokens,
        "perturbed_visual_samples": perturbed_visual_samples,
        "total": total,
        "sources": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
