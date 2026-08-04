"""Parameter-matched patch baseline for the native Slice-MoT experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fine_grain.byte_tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE
from fine_grain.native_vlm import MoTBlock, SliceMoTConfig, SliceMoTOutput, SwiGLU


@dataclass(frozen=True)
class PatchMoTConfig:
    layers: int = 6
    model_width: int = 512
    visual_tokens: int = 256
    attention_heads: int = 8
    visual_ffn_width: int = 1536
    text_ffn_width: int = 1536
    patch_ffn_width: int = 512
    vocabulary_size: int = VOCAB_SIZE
    image_size: int = 256
    patch_size: int = 16
    rope_base: float = 10_000.0
    visual_position_scale: float = 512.0
    dropout: float = 0.0
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        grid = self.image_size // self.patch_size
        if grid * grid != self.visual_tokens:
            raise ValueError("visual_tokens must equal the square patch-grid size")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if (self.model_width // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by four for 2D RoPE")

    def mot_config(self) -> SliceMoTConfig:
        return SliceMoTConfig(
            layers=self.layers,
            point_width=1,
            model_width=self.model_width,
            visual_slices=self.visual_tokens,
            attention_heads=self.attention_heads,
            visual_ffn_width=self.visual_ffn_width,
            text_ffn_width=self.text_ffn_width,
            vocabulary_size=self.vocabulary_size,
            rope_base=self.rope_base,
            visual_position_scale=self.visual_position_scale,
            dropout=self.dropout,
            activation_checkpointing=self.activation_checkpointing,
        )


class PatchStem(nn.Module):
    """Standard non-overlapping Conv2d patch embedding on a fixed token grid."""

    def __init__(self, config: PatchMoTConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_embed = nn.Conv2d(
            3,
            config.model_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        grid = self.patch_embed(images)
        batch, _, height, width = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        y = torch.linspace(-1.0, 1.0, height, device=grid.device, dtype=grid.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=grid.device, dtype=grid.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1).reshape(1, -1, 2)
        return tokens, coordinates.expand(batch, -1, -1)


class PatchVisualLayer(nn.Module):
    """Functional visual capacity matched to one Slice read/write layer."""

    def __init__(self, config: PatchMoTConfig) -> None:
        super().__init__()
        width = config.model_width
        self.grid_size = config.image_size // config.patch_size
        self.local_norm = nn.RMSNorm(width)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )
        self.ffn_norm = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, config.patch_ffn_width)

    def before_mot(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        grid = self.local_norm(tokens).transpose(1, 2).reshape(
            batch, tokens.shape[-1], self.grid_size, self.grid_size
        )
        return tokens + self.local(grid).flatten(2).transpose(1, 2)

    def after_mot(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.ffn(self.ffn_norm(tokens))


class PatchMoTVLM(nn.Module):
    """Patch-ViT visual control sharing the Slice model's language contract."""

    def __init__(self, config: PatchMoTConfig = PatchMoTConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.model_width, padding_idx=PAD_ID
        )
        self.stem = PatchStem(config)
        self.visual_layers = nn.ModuleList(
            [PatchVisualLayer(config) for _ in range(config.layers)]
        )
        mot_config = config.mot_config()
        self.mot_blocks = nn.ModuleList(
            [MoTBlock(mot_config) for _ in range(config.layers)]
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
        for layer in self.visual_layers:
            nn.init.normal_(layer.ffn.down.weight, std=residual_std)
        for block in self.mot_blocks:
            for expert in (block.visual_expert, block.text_expert):
                nn.init.normal_(expert.output.weight, std=residual_std)
                nn.init.normal_(expert.ffn.down.weight, std=residual_std)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _visual_step(
        visual: torch.Tensor,
        text: torch.Tensor,
        coordinates: torch.Tensor,
        visual_layer: PatchVisualLayer,
        mot_block: MoTBlock,
        prompt_length: int,
        text_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = visual_layer.before_mot(visual)
        visual, text = mot_block(
            text,
            prompt_length=prompt_length,
            text_valid=text_valid,
            visual=visual,
            visual_centroids=coordinates,
        )
        return visual_layer.after_mot(visual), text

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
    ) -> SliceMoTOutput:
        if prompt_ids.ndim != 2 or target_input_ids.ndim != 2:
            raise ValueError("token inputs must have shape [batch, sequence]")
        if prompt_ids.shape[0] != target_input_ids.shape[0]:
            raise ValueError("prompt and target batch sizes must match")
        if images is not None and images.shape[0] != prompt_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")
        if prompt_mask is None:
            prompt_mask = prompt_ids != PAD_ID
        if target_mask is None:
            target_mask = target_input_ids != PAD_ID
        text_valid = torch.cat((prompt_mask, target_mask), dim=1).bool()
        text = self.token_embedding(torch.cat((prompt_ids, target_input_ids), dim=1))
        text = text * text_valid.unsqueeze(-1)
        visual = coordinates = None
        if images is not None:
            visual, coordinates = self.stem(images)

        for visual_layer, mot_block in zip(self.visual_layers, self.mot_blocks):
            if visual is None:
                _, text = mot_block(
                    text,
                    prompt_length=prompt_ids.shape[1],
                    text_valid=text_valid,
                )
                continue

            def step(
                visual_states,
                text_states,
                positions=coordinates,
                layer=visual_layer,
                block=mot_block,
            ):
                return self._visual_step(
                    visual_states,
                    text_states,
                    positions,
                    layer,
                    block,
                    prompt_ids.shape[1],
                    text_valid,
                )

            if self.training and self.config.activation_checkpointing:
                visual, text = checkpoint(step, visual, text, use_reentrant=False)
            else:
                visual, text = step(visual, text)

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
            visual_field=None,
            diagnostics=[] if not return_state else [{"patch_tokens": visual.detach()}],
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
