"""Native persistent-field Slice-MoT vision-language model.

Vision and language keep separate parameters and persistent states. Transient
Slice states communicate with serialized text through global causal attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fine_grain.byte_tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE
from fine_grain.models import newton_schulz, sparse_deslice_weights
from fine_grain.slice_kernels import slice_assignment


@dataclass(frozen=True)
class SliceMoTConfig:
    layers: int = 6
    point_width: int = 256
    model_width: int = 512
    visual_slices: int = 256
    attention_heads: int = 8
    visual_ffn_width: int = 1536
    text_ffn_width: int = 1536
    vocabulary_size: int = VOCAB_SIZE
    tile_points: int = 4096
    rope_base: float = 10_000.0
    visual_position_scale: float = 512.0
    dropout: float = 0.0
    assignment_backend: str = "auto"
    activation_checkpointing: bool = True
    standardize_thin_detail: bool = False
    point_adaptive_temperature: bool = False
    gumbel_assignment: bool = False
    stiefel_slices: bool = True
    deslice_topk: int = 0

    def __post_init__(self) -> None:
        values = (
            self.layers,
            self.point_width,
            self.model_width,
            self.visual_slices,
            self.attention_heads,
            self.visual_ffn_width,
            self.text_ffn_width,
            self.tile_points,
        )
        if min(values) < 1:
            raise ValueError("all dimensions must be positive")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if (self.model_width // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by four for 2D RoPE")
        if self.assignment_backend not in {"auto", "torch", "triton"}:
            raise ValueError("assignment_backend must be auto, torch, or triton")
        if self.deslice_topk < 0 or self.deslice_topk > self.visual_slices:
            raise ValueError("deslice_topk must be between zero and visual_slices")


@dataclass
class VisualField:
    points: torch.Tensor
    coordinates: torch.Tensor
    grid_shape: tuple[int, int]


@dataclass
class SliceMoTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    visual_field: VisualField | None = None
    diagnostics: list[dict[str, torch.Tensor]] = field(default_factory=list)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class PointFieldStem(nn.Module):
    """Create one persistent state per RGB pixel without downsampling."""

    def __init__(self, point_width: int) -> None:
        super().__init__()
        self.point_width = point_width
        self.projection = nn.Linear(5, point_width)
        self.local = nn.Sequential(
            nn.Conv2d(point_width, point_width, 3, padding=1, groups=point_width),
            nn.Conv2d(point_width, point_width, 1),
        )

    @staticmethod
    def coordinates(
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2).expand(
            batch, -1, -1
        )

    def forward(self, images: torch.Tensor) -> VisualField:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        batch, _, height, width = images.shape
        coordinates = self.coordinates(
            batch,
            height,
            width,
            device=images.device,
            dtype=images.dtype,
        )
        rgb = images.permute(0, 2, 3, 1).reshape(batch, height * width, 3)
        points = self.projection(torch.cat((rgb, coordinates), dim=-1))
        grid = points.transpose(1, 2).reshape(batch, self.point_width, height, width)
        points = points + self.local(grid).flatten(2).transpose(1, 2)
        return VisualField(points, coordinates, (height, width))


class PersistentSliceLayer(nn.Module):
    """Read transient Slice states and write them back to a persistent point field."""

    def __init__(self, config: SliceMoTConfig) -> None:
        super().__init__()
        point_width = config.point_width
        model_width = config.model_width
        self.point_width = point_width
        self.model_width = model_width
        self.visual_slices = config.visual_slices
        self.tile_points = config.tile_points
        self.assignment_backend = config.assignment_backend
        self.standardize_thin_detail = config.standardize_thin_detail
        self.point_adaptive_temperature = config.point_adaptive_temperature
        self.gumbel_assignment = config.gumbel_assignment
        self.stiefel_slices = config.stiefel_slices
        self.deslice_topk = config.deslice_topk

        self.pre_norm = nn.RMSNorm(point_width)
        self.pre_local = nn.Sequential(
            nn.Conv2d(point_width, point_width, 3, padding=1, groups=point_width),
            nn.Conv2d(point_width, point_width, 1),
        )
        self.assignment_key = nn.Linear(point_width, point_width, bias=False)
        nn.init.orthogonal_(self.assignment_key.weight)
        self.slice_queries = nn.Parameter(torch.empty(config.visual_slices, point_width))
        nn.init.normal_(self.slice_queries)
        if self.point_adaptive_temperature:
            self.temperature_projection = nn.Sequential(
                nn.Linear(point_width, config.visual_slices),
                nn.GELU(),
                nn.Linear(config.visual_slices, 1),
                nn.GELU(),
            )
            self.temperature_bias = nn.Parameter(torch.tensor(0.5))
        else:
            self.temperature_projection = None
            self.register_parameter("temperature_bias", None)
        self.point_value = nn.Linear(point_width, model_width, bias=False)
        self.workspace_to_point = nn.Linear(model_width, point_width, bias=False)
        self.point_ffn_norm = nn.RMSNorm(point_width)
        self.point_ffn = SwiGLU(point_width, point_width * 3)
        self.post_norm = nn.RMSNorm(point_width)
        self.post_local = nn.Sequential(
            nn.Conv2d(point_width, point_width, 3, padding=1, groups=point_width),
            nn.Conv2d(point_width, point_width, 1),
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def local_update(self, field: VisualField, *, post: bool = False) -> VisualField:
        height, width = field.grid_shape
        norm = self.post_norm if post else self.pre_norm
        local = self.post_local if post else self.pre_local
        normalized = norm(field.points)
        grid = normalized.transpose(1, 2).reshape(
            field.points.shape[0], self.point_width, height, width
        )
        points = field.points + local(grid).flatten(2).transpose(1, 2)
        return VisualField(points, field.coordinates, field.grid_shape)

    def _fused_assignment_weight(self) -> torch.Tensor:
        """Fuse point projection and Slice queries into one point-domain GEMM."""
        return self.slice_queries @ self.assignment_key.weight

    def _weights(
        self, points: torch.Tensor, fused_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.pre_norm(points)
        if self.temperature_projection is None and not (
            self.training and self.gumbel_assignment
        ):
            weights = slice_assignment(
                normalized, fused_weight, backend=self.assignment_backend
            )
        else:
            logits = F.linear(normalized, fused_weight) / math.sqrt(self.point_width)
            if self.training and self.gumbel_assignment:
                uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                logits = logits - torch.log(-torch.log(uniform))
            if self.temperature_projection is None:
                temperature = 1.0
            else:
                temperature = (
                    self.temperature_projection(normalized) + self.temperature_bias
                ).clamp_min(0.01)
            weights = F.softmax(logits / temperature, dim=-1)
        return weights, normalized

    def read(
        self, field: VisualField, *, collect_diagnostics: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = field.points.shape[0]
        numerator = torch.zeros(
            batch,
            self.visual_slices,
            self.point_width,
            device=field.points.device,
            dtype=torch.float32,
        )
        coordinate_numerator = torch.zeros(
            batch,
            self.visual_slices,
            2,
            device=field.points.device,
            dtype=torch.float32,
        )
        mass = torch.zeros(
            batch,
            self.visual_slices,
            device=field.points.device,
            dtype=torch.float32,
        )
        global_numerator = torch.zeros(
            batch,
            self.point_width,
            device=field.points.device,
            dtype=torch.float32,
        )
        entropy_sum = None
        if collect_diagnostics:
            entropy_sum = torch.zeros(
                (), device=field.points.device, dtype=torch.float32
            )
        fused_weight = self._fused_assignment_weight()

        for start in range(0, field.points.shape[1], self.tile_points):
            end = min(field.points.shape[1], start + self.tile_points)
            weights, normalized = self._weights(
                field.points[:, start:end], fused_weight
            )
            weights_float = weights.float()
            numerator = numerator + torch.einsum(
                "bnm,bnc->bmc", weights_float, normalized.float()
            )
            coordinate_numerator = coordinate_numerator + torch.einsum(
                "bnm,bnd->bmd",
                weights_float,
                field.coordinates[:, start:end].float(),
            )
            mass = mass + weights_float.sum(dim=1)
            global_numerator = global_numerator + normalized.float().sum(dim=1)
            if entropy_sum is not None:
                entropy_sum = entropy_sum - (
                    weights_float * weights_float.clamp_min(1e-12).log()
                ).sum()

        raw_slices = numerator / mass.clamp_min(1e-6).unsqueeze(-1)
        if self.standardize_thin_detail:
            # A one-pixel curve occupies O(sqrt(N)) points. Global averaging
            # attenuates it by 1/sqrt(N), so restore that statistical scale
            # while retaining the global low-frequency component.
            global_mean = (
                global_numerator / field.points.shape[1]
            ).unsqueeze(1)
            detail_gain = math.sqrt(field.points.shape[1])
            raw_slices = global_mean + detail_gain * (raw_slices - global_mean)
        # Linear associativity: aggregate in point width, then project only M states.
        slices = self.point_value(raw_slices.to(field.points))
        if self.stiefel_slices:
            directions = slices.transpose(1, 2)
            magnitude = directions.norm(dim=1, keepdim=True).clamp_min(1e-6)
            directions = newton_schulz(directions)
            slices = (directions * magnitude).transpose(1, 2)
        centroids = coordinate_numerator / mass.clamp_min(1e-6).unsqueeze(-1)
        self.last_diagnostics = {}
        if entropy_sum is not None:
            probability = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            effective = torch.exp(
                -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            )
            entropy = entropy_sum / (field.points.shape[0] * field.points.shape[1])
            self.last_diagnostics = {
                # Keep diagnostics on device. Converting to Python scalars here
                # would synchronize CUDA once per visual layer and training step.
                "assignment_entropy": entropy.detach(),
                "effective_slices": effective.mean().detach(),
                "point_count": torch.as_tensor(
                    field.points.shape[1], device=field.points.device
                ),
            }
        return slices.to(field.points), centroids.to(field.points)

    def write(self, field: VisualField, workspace: torch.Tensor) -> VisualField:
        # Project only M Slice states before scattering to N points.
        projected = self.workspace_to_point(workspace.to(field.points))
        chunks = []
        fused_weight = self._fused_assignment_weight()
        for start in range(0, field.points.shape[1], self.tile_points):
            end = min(field.points.shape[1], start + self.tile_points)
            points = field.points[:, start:end]
            weights, _ = self._weights(points, fused_weight)
            if self.deslice_topk:
                weights = sparse_deslice_weights(weights, topk=self.deslice_topk)
            message = torch.einsum("bnm,bmc->bnc", weights, projected)
            updated = points + message
            updated = updated + self.point_ffn(self.point_ffn_norm(updated))
            chunks.append(updated)
        updated_field = VisualField(
            torch.cat(chunks, dim=1), field.coordinates, field.grid_shape
        )
        return self.local_update(updated_field, post=True)


def _apply_axis_rope(
    values: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Apply rotary embedding to one even-width axis."""
    axis_width = values.shape[-1]
    if axis_width % 2:
        raise ValueError("RoPE axis width must be even")
    inverse = 1.0 / (
        base
        ** (
            torch.arange(0, axis_width, 2, device=values.device, dtype=torch.float32)
            / axis_width
        )
    )
    angles = positions.float()[:, None, :, None] * inverse[None, None, None, :]
    cosine = angles.cos().to(values)
    sine = angles.sin().to(values)
    pairs = values.reshape(*values.shape[:-1], axis_width // 2, 2)
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
    )
    return rotated.flatten(-2)


def apply_2d_rope(
    values: torch.Tensor,
    y_positions: torch.Tensor,
    x_positions: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    """Split each attention head equally between y and x rotary axes."""
    y_values, x_values = values.chunk(2, dim=-1)
    return torch.cat(
        (
            _apply_axis_rope(y_values, y_positions, base=base),
            _apply_axis_rope(x_values, x_positions, base=base),
        ),
        dim=-1,
    )


class ModalityExpert(nn.Module):
    """All non-embedding parameters for one MoT modality."""

    def __init__(self, width: int, ffn_width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.attention_norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_width)

    def qkv(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(states)
        shape = (states.shape[0], states.shape[1], self.heads, self.head_width)
        query = self.query(normalized).reshape(shape).transpose(1, 2)
        key = self.key(normalized).reshape(shape).transpose(1, 2)
        value = self.value(normalized).reshape(shape).transpose(1, 2)
        return query, key, value

    def finish_attention(self, residual: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        attended = attended.transpose(1, 2).reshape_as(residual)
        states = residual + self.output(attended)
        return states + self.ffn(self.ffn_norm(states))


class MoTBlock(nn.Module):
    """Modality-specific experts connected by serialized global causal attention."""

    def __init__(self, config: SliceMoTConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_width = config.model_width // config.attention_heads
        self.rope_base = config.rope_base
        self.visual_position_scale = config.visual_position_scale
        self.dropout = config.dropout
        self.visual_expert = ModalityExpert(
            config.model_width, config.visual_ffn_width, config.attention_heads
        )
        self.text_expert = ModalityExpert(
            config.model_width, config.text_ffn_width, config.attention_heads
        )

    @staticmethod
    def _attention_mask(
        query_positions: torch.Tensor,
        key_valid: torch.Tensor,
        key_count: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key_positions = torch.arange(key_count, device=query_positions.device)
        allowed = key_positions.reshape(1, 1, 1, -1) <= query_positions.reshape(
            1, 1, -1, 1
        )
        allowed = allowed & key_valid[:, None, None, :]
        mask = torch.zeros(allowed.shape, device=query_positions.device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._attention_mask(
            query_positions, key_valid, key.shape[2], query.dtype
        )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def forward(
        self,
        text: torch.Tensor,
        *,
        prompt_length: int,
        text_valid: torch.Tensor,
        visual: torch.Tensor | None = None,
        visual_centroids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch, text_length, _ = text.shape
        target_length = text_length - prompt_length
        visual_length = 0 if visual is None else visual.shape[1]
        total_length = text_length + visual_length

        text_query, text_key, text_value = self.text_expert.qkv(text)
        prompt_key, target_key = text_key.split((prompt_length, target_length), dim=2)
        prompt_value, target_value = text_value.split(
            (prompt_length, target_length), dim=2
        )

        prompt_positions = torch.arange(prompt_length, device=text.device)
        target_positions = torch.arange(
            prompt_length + visual_length,
            total_length,
            device=text.device,
        )
        text_positions = torch.cat((prompt_positions, target_positions))
        text_x = text_positions.reshape(1, -1).expand(batch, -1)
        text_y = torch.zeros_like(text_x)
        text_query = apply_2d_rope(
            text_query, text_y, text_x, base=self.rope_base
        )
        text_key = apply_2d_rope(text_key, text_y, text_x, base=self.rope_base)
        prompt_key, target_key = text_key.split((prompt_length, target_length), dim=2)

        if visual is None:
            key = text_key
            value = text_value
            key_valid = text_valid
            attended_text = self._attend(
                text_query, key, value, text_positions, key_valid
            )
            text = self.text_expert.finish_attention(text, attended_text)
            return None, text * text_valid.unsqueeze(-1)

        if visual_centroids is None:
            raise ValueError("visual_centroids are required with visual states")
        visual_query, visual_key, visual_value = self.visual_expert.qkv(visual)
        visual_y = (visual_centroids[..., 0] + 1.0) * (
            self.visual_position_scale / 2.0
        )
        visual_x = (visual_centroids[..., 1] + 1.0) * (
            self.visual_position_scale / 2.0
        )
        visual_query = apply_2d_rope(
            visual_query, visual_y, visual_x, base=self.rope_base
        )
        visual_key = apply_2d_rope(
            visual_key, visual_y, visual_x, base=self.rope_base
        )

        key = torch.cat((prompt_key, visual_key, target_key), dim=2)
        value = torch.cat((prompt_value, visual_value, target_value), dim=2)
        prompt_valid, target_valid = text_valid.split(
            (prompt_length, target_length), dim=1
        )
        visual_valid = torch.ones(
            batch, visual_length, device=text.device, dtype=torch.bool
        )
        key_valid = torch.cat((prompt_valid, visual_valid, target_valid), dim=1)

        visual_positions = torch.arange(
            prompt_length,
            prompt_length + visual_length,
            device=text.device,
        )
        attended_visual = self._attend(
            visual_query, key, value, visual_positions, key_valid
        )
        attended_text = self._attend(
            text_query, key, value, text_positions, key_valid
        )
        visual = self.visual_expert.finish_attention(visual, attended_visual)
        text = self.text_expert.finish_attention(text, attended_text)
        return visual, text * text_valid.unsqueeze(-1)


class SliceMoTVLM(nn.Module):
    """Six-layer native VLM with parallel persistent vision and language streams."""

    def __init__(self, config: SliceMoTConfig = SliceMoTConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.model_width, padding_idx=PAD_ID
        )
        self.stem = PointFieldStem(config.point_width)
        self.visual_layers = nn.ModuleList(
            [PersistentSliceLayer(config) for _ in range(config.layers)]
        )
        self.mot_blocks = nn.ModuleList(
            [MoTBlock(config) for _ in range(config.layers)]
        )
        self.final_norm = nn.RMSNorm(config.model_width)
        self.lm_head = nn.Linear(
            config.model_width, config.vocabulary_size, bias=False
        )
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()
        for visual_layer in self.visual_layers:
            nn.init.orthogonal_(visual_layer.assignment_key.weight)
            nn.init.normal_(visual_layer.slice_queries)
            nn.init.normal_(visual_layer.point_ffn.down.weight, std=residual_std)
        for block in self.mot_blocks:
            for expert in (block.visual_expert, block.text_expert):
                nn.init.normal_(expert.output.weight, std=residual_std)
                nn.init.normal_(expert.ffn.down.weight, std=residual_std)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _visual_step(
        points: torch.Tensor,
        text: torch.Tensor,
        coordinates: torch.Tensor,
        grid_shape: tuple[int, int],
        visual_layer: PersistentSliceLayer,
        mot_block: MoTBlock,
        prompt_length: int,
        text_valid: torch.Tensor,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        field = VisualField(points, coordinates, grid_shape)
        field = visual_layer.local_update(field)
        slices, centroids = visual_layer.read(
            field, collect_diagnostics=collect_diagnostics
        )
        slices, text = mot_block(
            text,
            prompt_length=prompt_length,
            text_valid=text_valid,
            visual=slices,
            visual_centroids=centroids,
        )
        field = visual_layer.write(field, slices)
        return field.points, text

    def forward(
        self,
        *,
        prompt_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_state: bool = False,
        collect_diagnostics: bool = True,
    ) -> SliceMoTOutput:
        if prompt_ids.ndim != 2 or target_input_ids.ndim != 2:
            raise ValueError("token inputs must have shape [batch, sequence]")
        if prompt_ids.shape[0] != target_input_ids.shape[0]:
            raise ValueError("prompt and target batch sizes must match")
        batch = prompt_ids.shape[0]
        if images is not None and images.shape[0] != batch:
            raise ValueError("image and text batch sizes must match")

        if prompt_mask is None:
            prompt_mask = prompt_ids != PAD_ID
        if target_mask is None:
            target_mask = target_input_ids != PAD_ID
        text_valid = torch.cat((prompt_mask, target_mask), dim=1).bool()
        text = self.token_embedding(torch.cat((prompt_ids, target_input_ids), dim=1))
        text = text * text_valid.unsqueeze(-1)
        field_state = None if images is None else self.stem(images)
        diagnostics = []

        for visual_layer, mot_block in zip(self.visual_layers, self.mot_blocks):
            if field_state is None:
                _, text = mot_block(
                    text,
                    prompt_length=prompt_ids.shape[1],
                    text_valid=text_valid,
                )
                continue
            def step(
                points,
                text_states,
                coordinates=field_state.coordinates,
                grid_shape=field_state.grid_shape,
                layer=visual_layer,
                block=mot_block,
            ):
                return self._visual_step(
                    points,
                    text_states,
                    coordinates,
                    grid_shape,
                    layer,
                    block,
                    prompt_ids.shape[1],
                    text_valid,
                    collect_diagnostics,
                )
            if self.training and self.config.activation_checkpointing:
                points, text = checkpoint(
                    step, field_state.points, text, use_reentrant=False
                )
            else:
                points, text = step(field_state.points, text)
            field_state = VisualField(
                points, field_state.coordinates, field_state.grid_shape
            )
            if collect_diagnostics:
                diagnostics.append(dict(visual_layer.last_diagnostics))

        target_states = self.final_norm(text[:, prompt_ids.shape[1] :])
        logits = self.lm_head(target_states)
        loss = None
        if labels is not None:
            if labels.shape != target_input_ids.shape:
                raise ValueError("labels must match target_input_ids shape")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SliceMoTOutput(
            logits=logits,
            loss=loss,
            visual_field=field_state if return_state else None,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt_ids: torch.Tensor,
        images: torch.Tensor | None,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (prompt_ids.shape[0], 1),
            BOS_ID,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        finished = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            output = self(
                prompt_ids=prompt_ids,
                target_input_ids=generated,
                images=images,
            )
            token = output.logits[:, -1].argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, EOS_ID), token)
            generated = torch.cat((generated, token[:, None]), dim=1)
            finished = finished | (token == EOS_ID)
            if bool(finished.all()):
                break
        return generated[:, 1:]
