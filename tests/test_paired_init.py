"""The paired arms must begin with identical shared parameters."""
from __future__ import annotations

import torch

from fine_grain.native_vlm import SliceMoTConfig, SliceMoTVLM
from fine_grain.paired_init import COMMON_PREFIXES, initialize_paired_common
from fine_grain.patch_vlm import PatchMoTConfig, PatchMoTVLM


def test_shared_parameters_are_bit_identical_after_paired_initialization():
    torch.manual_seed(1)
    slice_model = SliceMoTVLM(SliceMoTConfig())
    torch.manual_seed(1)
    patch_model = PatchMoTVLM(PatchMoTConfig())
    initialize_paired_common(slice_model, 20260805)
    initialize_paired_common(patch_model, 20260805)
    slice_parameters = dict(slice_model.named_parameters())
    patch_parameters = dict(patch_model.named_parameters())
    common = sorted(
        name
        for name in slice_parameters.keys() & patch_parameters.keys()
        if name.startswith(COMMON_PREFIXES)
    )
    assert common
    for name in common:
        torch.testing.assert_close(
            slice_parameters[name], patch_parameters[name], atol=0.0, rtol=0.0
        )
