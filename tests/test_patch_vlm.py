"""Fairness and causal tests for the parameter-matched Patch-MoT control."""
from __future__ import annotations

from dataclasses import replace

import torch

from fine_grain.byte_tokenizer import BOS_ID, EOS_ID, OCR_ID
from fine_grain.native_vlm import SliceMoTConfig, SliceMoTVLM
from fine_grain.patch_vlm import PatchMoTConfig, PatchMoTVLM


def tiny_config(*, checkpointing: bool = False) -> PatchMoTConfig:
    return PatchMoTConfig(
        layers=2,
        model_width=32,
        visual_tokens=16,
        attention_heads=4,
        visual_ffn_width=64,
        text_ffn_width=64,
        patch_ffn_width=32,
        image_size=16,
        patch_size=4,
        activation_checkpointing=checkpointing,
    )


def inputs(batch: int = 1):
    prompt = torch.tensor([[BOS_ID, OCR_ID]]).expand(batch, -1).clone()
    target = torch.tensor([[BOS_ID, 100, 101]]).expand(batch, -1).clone()
    labels = torch.tensor([[100, 101, EOS_ID]]).expand(batch, -1).clone()
    images = torch.rand(batch, 3, 13, 19)
    return prompt, target, labels, images


def test_registered_patch_baseline_is_parameter_matched():
    slice_count = SliceMoTVLM(SliceMoTConfig()).parameter_count()
    patch_count = PatchMoTVLM(PatchMoTConfig()).parameter_count()
    relative_gap = abs(slice_count - patch_count) / slice_count
    assert relative_gap < 0.002, (slice_count, patch_count, relative_gap)
    assert PatchMoTConfig().visual_tokens == SliceMoTConfig().visual_slices


def test_patch_initial_loss_and_rectangular_image_path():
    torch.manual_seed(11)
    model = PatchMoTVLM(tiny_config()).eval()
    prompt, target, labels, images = inputs(batch=2)
    with torch.no_grad():
        output = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=images,
            labels=labels,
            return_state=True,
        )
    assert output.logits.shape == (2, 3, model.config.vocabulary_size)
    assert 3.0 < float(output.loss) < 10.0
    assert output.diagnostics[0]["patch_tokens"].shape == (2, 16, 32)


def test_patch_future_target_cannot_change_visual_or_earlier_logits():
    torch.manual_seed(12)
    model = PatchMoTVLM(tiny_config()).eval()
    prompt, target, _, images = inputs()
    changed = target.clone()
    changed[:, -1] = 177
    with torch.no_grad():
        baseline = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=images,
            return_state=True,
        )
        counterfactual = model(
            prompt_ids=prompt,
            target_input_ids=changed,
            images=images,
            return_state=True,
        )
    torch.testing.assert_close(
        baseline.logits[:, :-1], counterfactual.logits[:, :-1], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        baseline.diagnostics[0]["patch_tokens"],
        counterfactual.diagnostics[0]["patch_tokens"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_patch_loss_reaches_pixels_and_every_checkpointed_layer():
    torch.manual_seed(13)
    model = PatchMoTVLM(tiny_config(checkpointing=True)).train()
    prompt, target, labels, images = inputs()
    images.requires_grad_(True)
    output = model(
        prompt_ids=prompt,
        target_input_ids=target,
        images=images,
        labels=labels,
    )
    output.loss.backward()
    assert images.grad is not None and float(images.grad.abs().sum()) > 0
    # Every layer's pre-MoT visual state is consumed by text K/V. The final
    # post-MoT visual FFN is a standard causal-depth boundary: no later text
    # layer exists to consume it, which is also true for the Slice control.
    assert all(layer.local[1].weight.grad is not None for layer in model.visual_layers)
    assert all(
        block.visual_expert.key.weight.grad is not None for block in model.mot_blocks
    )


def test_patch_text_only_contract_matches_visual_contract():
    model = PatchMoTVLM(replace(tiny_config(), layers=1)).eval()
    prompt, target, _, images = inputs()
    with torch.no_grad():
        text_only = model(prompt_ids=prompt, target_input_ids=target)
        visual = model(
            prompt_ids=prompt, target_input_ids=target, images=images
        )
    assert text_only.logits.shape == visual.logits.shape
