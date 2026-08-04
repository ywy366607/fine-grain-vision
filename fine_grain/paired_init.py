"""Reproducible common-parameter initialization for paired architectures."""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from fine_grain.byte_tokenizer import PAD_ID


COMMON_PREFIXES = ("token_embedding.", "mot_blocks.", "final_norm.", "lm_head.")


def _parameter_seed(experiment_seed: int, name: str) -> int:
    digest = hashlib.blake2b(
        f"{experiment_seed}:{name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def initialize_paired_common(model: nn.Module, experiment_seed: int) -> None:
    """Make every shared name/shape bit-identical across Slice and Patch arms."""
    layers = model.config.layers
    seen: set[int] = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith(COMMON_PREFIXES) or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim == 1:
                parameter.fill_(1.0)
                continue
            standard_deviation = 0.02
            if name.endswith("output.weight") or name.endswith("ffn.down.weight"):
                standard_deviation /= math.sqrt(2 * layers)
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(_parameter_seed(experiment_seed, name))
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                * standard_deviation
            )
        model.token_embedding.weight[PAD_ID].zero_()
