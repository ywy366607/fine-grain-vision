"""Mechanism tests for the native persistent-field Slice-MoT VLM."""
from __future__ import annotations

from dataclasses import replace

import torch

from fine_grain.byte_tokenizer import BOS_ID, ByteTokenizer, EOS_ID, OCR_ID
from fine_grain.native_vlm import SliceMoTConfig, SliceMoTVLM
from fine_grain.slice_kernels import slice_assignment


def tiny_config(*, layers: int = 2, assignment_backend: str = "auto") -> SliceMoTConfig:
    return SliceMoTConfig(
        layers=layers,
        point_width=16,
        model_width=32,
        visual_slices=4,
        attention_heads=4,
        visual_ffn_width=64,
        text_ffn_width=64,
        tile_points=32,
        assignment_backend=assignment_backend,
        activation_checkpointing=False,
    )


def inputs(batch: int = 2):
    prompt = torch.tensor([[BOS_ID, OCR_ID]]).expand(batch, -1).clone()
    target = torch.tensor([[BOS_ID, 100, 101]]).expand(batch, -1).clone()
    prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    images = torch.rand(batch, 3, 8, 10)
    return prompt, target, prompt_mask, target_mask, images


def test_byte_tokenizer_round_trip_and_teacher_forcing():
    tokenizer = ByteTokenizer()
    text = "ASCII and 中文"
    assert tokenizer.decode(tokenizer.encode(text, add_eos=True)) == text
    batch = tokenizer.teacher_forcing_batch(
        [tokenizer.task_prompt("ocr"), tokenizer.task_prompt("ocr")],
        ["ab", "x"],
    )
    assert batch.target_input_ids.tolist() == [
        [BOS_ID, ord("a") + 3, ord("b") + 3],
        [BOS_ID, ord("x") + 3, 0],
    ]
    assert batch.target_labels.tolist() == [
        [ord("a") + 3, ord("b") + 3, EOS_ID],
        [ord("x") + 3, EOS_ID, -100],
    ]


def test_native_vlm_shape_loss_and_modality_parameters_are_untied():
    torch.manual_seed(1)
    model = SliceMoTVLM(tiny_config())
    prompt, target, prompt_mask, target_mask, images = inputs()
    labels = torch.tensor([[100, 101, EOS_ID], [100, 101, EOS_ID]])
    output = model(
        prompt_ids=prompt,
        target_input_ids=target,
        prompt_mask=prompt_mask,
        target_mask=target_mask,
        images=images,
        labels=labels,
    )
    assert output.logits.shape == (2, 3, model.config.vocabulary_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert len(output.diagnostics) == model.config.layers
    assert all(
        isinstance(value, torch.Tensor)
        for diagnostics in output.diagnostics
        for value in diagnostics.values()
    )
    block = model.mot_blocks[0]
    assert block.visual_expert.query.weight is not block.text_expert.query.weight
    assert block.visual_expert.ffn.up.weight is not block.text_expert.ffn.up.weight


def test_future_target_cannot_change_earlier_logits_or_visual_field():
    torch.manual_seed(2)
    model = SliceMoTVLM(tiny_config()).eval()
    prompt, target, prompt_mask, target_mask, images = inputs(batch=1)
    changed = target.clone()
    changed[:, -1] = 177
    with torch.no_grad():
        baseline = model(
            prompt_ids=prompt,
            target_input_ids=target,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
            images=images,
            return_state=True,
        )
        counterfactual = model(
            prompt_ids=prompt,
            target_input_ids=changed,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
            images=images,
            return_state=True,
        )
    torch.testing.assert_close(
        baseline.logits[:, :-1], counterfactual.logits[:, :-1], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        baseline.visual_field.points,
        counterfactual.visual_field.points,
        atol=1e-6,
        rtol=1e-6,
    )


def test_prompt_changes_visual_field_but_target_does_not():
    torch.manual_seed(3)
    model = SliceMoTVLM(tiny_config()).eval()
    prompt, target, prompt_mask, target_mask, images = inputs(batch=1)
    changed_prompt = prompt.clone()
    changed_prompt[:, 1] = OCR_ID + 1
    with torch.no_grad():
        baseline = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=images,
            return_state=True,
        )
        changed = model(
            prompt_ids=changed_prompt,
            target_input_ids=target,
            images=images,
            return_state=True,
        )
    assert not torch.allclose(baseline.visual_field.points, changed.visual_field.points)


def test_target_loss_reaches_rgb_and_slice_assignment():
    torch.manual_seed(4)
    model = SliceMoTVLM(tiny_config(layers=1))
    prompt, target, prompt_mask, target_mask, images = inputs(batch=1)
    images.requires_grad_(True)
    labels = torch.tensor([[100, 101, EOS_ID]])
    output = model(
        prompt_ids=prompt,
        target_input_ids=target,
        images=images,
        labels=labels,
    )
    output.loss.backward()
    assert images.grad is not None and float(images.grad.abs().sum()) > 0
    queries = model.visual_layers[0].slice_queries
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0


def test_activation_checkpointed_visual_step_preserves_gradients():
    torch.manual_seed(9)
    config = replace(tiny_config(layers=2), activation_checkpointing=True)
    model = SliceMoTVLM(config).train()
    prompt, target, _, _, images = inputs(batch=1)
    images.requires_grad_(True)
    output = model(
        prompt_ids=prompt,
        target_input_ids=target,
        images=images,
        labels=torch.tensor([[100, 101, EOS_ID]]),
    )
    output.loss.backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert all(layer.slice_queries.grad is not None for layer in model.visual_layers)


def test_text_only_and_rectangular_visual_paths():
    torch.manual_seed(5)
    model = SliceMoTVLM(tiny_config(layers=1)).eval()
    prompt, target, _, _, images = inputs(batch=1)
    with torch.no_grad():
        text_only = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=None,
        )
        visual = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=images,
            return_state=True,
        )
    assert text_only.logits.shape == visual.logits.shape
    assert visual.visual_field.grid_shape == (8, 10)
    assert visual.visual_field.points.shape[1] == 80


def test_registered_config_parameter_count():
    model = SliceMoTVLM(SliceMoTConfig())
    count = model.parameter_count()
    assert 40_000_000 <= count <= 50_000_000, count
    assert model.config.point_width * 2 == model.config.model_width
    assert len({id(layer.slice_queries) for layer in model.visual_layers}) == 6
    assert len({id(layer.assignment_key.weight) for layer in model.visual_layers}) == 6
    assert len({id(layer.workspace_to_point.weight) for layer in model.visual_layers}) == 6


def test_initial_loss_is_on_a_trainable_scale():
    torch.manual_seed(10)
    model = SliceMoTVLM(tiny_config(layers=2)).eval()
    prompt, target, _, _, images = inputs(batch=2)
    labels = torch.tensor([[100, 101, EOS_ID], [100, 101, EOS_ID]])
    with torch.no_grad():
        loss = model(
            prompt_ids=prompt,
            target_input_ids=target,
            images=images,
            labels=labels,
        ).loss
    assert 3.0 < float(loss) < 10.0, float(loss)


def test_slice_assignment_torch_rows_sum_to_one():
    points = torch.randn(2, 7, 16)
    weight = torch.randn(4, 16)
    assignment = slice_assignment(points, weight, backend="torch")
    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.ones(2, 7), atol=1e-6, rtol=1e-6
    )


def test_triton_assignment_matches_torch_forward_and_backward():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(6)
    points_torch = torch.randn(2, 37, 32, device="cuda", requires_grad=True)
    weight_torch = torch.randn(16, 32, device="cuda", requires_grad=True)
    points_triton = points_torch.detach().clone().requires_grad_(True)
    weight_triton = weight_torch.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 37, 16, device="cuda")

    torch_output = slice_assignment(points_torch, weight_torch, backend="torch")
    triton_output = slice_assignment(points_triton, weight_triton, backend="triton")
    torch.testing.assert_close(triton_output, torch_output, atol=2e-5, rtol=2e-5)
    (torch_output * upstream).sum().backward()
    (triton_output * upstream).sum().backward()
    torch.testing.assert_close(
        points_triton.grad, points_torch.grad, atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        weight_triton.grad, weight_torch.grad, atol=3e-5, rtol=3e-5
    )


def test_triton_assignment_handles_tail_slices_and_production_width():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(7)
    points = torch.randn(1, 19, 256, device="cuda")
    weight = torch.randn(193, 256, device="cuda")

    expected = slice_assignment(points, weight, backend="torch")
    actual = slice_assignment(points, weight, backend="triton")

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(1, 19, device="cuda"), atol=2e-6, rtol=2e-6
    )


def test_triton_bfloat16_assignment_and_native_vlm_backward():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(8)
    points = torch.randn(2, 71, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(193, 128, device="cuda", dtype=torch.bfloat16)
    expected = slice_assignment(points, weight, backend="torch")
    actual = slice_assignment(points, weight, backend="triton")
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)

    model = SliceMoTVLM(
        tiny_config(layers=1, assignment_backend="triton")
    ).cuda().to(torch.bfloat16)
    prompt, target, _, _, images = inputs(batch=1)
    output = model(
        prompt_ids=prompt.cuda(),
        target_input_ids=target.cuda(),
        images=images.cuda().to(torch.bfloat16),
        labels=torch.tensor([[100, 101, EOS_ID]], device="cuda"),
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert model.visual_layers[0].slice_queries.grad is not None
