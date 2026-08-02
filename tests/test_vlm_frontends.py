"""Unit tests for A/B/C vision frontends — drives real fine_grain.frontends."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.frontends import (  # noqa: E402
    HybridFrontend,
    PatchFrontend,
    SliceFrontend,
    build_frontend,
    patch_token_count,
    suggest_T_grid,
)


def test_patch_token_count_and_T_grid():
    assert patch_token_count(32, 4) == 64
    assert patch_token_count(64, 16) == 16
    g = suggest_T_grid(32, 4)
    assert g[0] == 64
    assert 32 in g


def test_A_shape_and_pool():
    d_llm = 128
    fe = PatchFrontend(d_llm, res=32, patch=4, dim=64, depth=1, T=64)
    y = fe(torch.rand(2, 3, 32, 32))
    assert y.tokens.shape == (2, 64, d_llm)
    fe16 = PatchFrontend(d_llm, res=32, patch=4, dim=64, depth=1, T=16)
    y2 = fe16(torch.rand(2, 3, 32, 32))
    assert y2.tokens.shape == (2, 16, d_llm)


def test_B_multi_T_and_st_topk():
    d_llm = 96
    for T in (64, 32):
        fe = SliceFrontend(d_llm, res=32, T=T, dim=64, depth=1, deslice_topk=2, stiefel=True)
        fe.eval()
        y = fe(torch.rand(2, 3, 32, 32))
        assert y.tokens.shape == (2, T, d_llm), y.tokens.shape
        assert y.meta["kind"] == "B"
        assert y.meta["deslice_topk"] == 2
        assert y.meta["stiefel_ns"] is True
        # support available after eval forward
        if y.meta.get("slot"):
            assert y.meta["slot"].get("support", T) <= T + 1e-3


def test_C_budget_split():
    d_llm = 64
    fe = HybridFrontend(d_llm, res=32, patch=8, T_patch=16, T_slice=16, depth=1)
    y = fe(torch.rand(1, 3, 32, 32))
    assert y.tokens.shape[1] == y.meta["T_patch"] + y.meta["T_slice"]
    assert y.meta["kind"] == "C"
    assert y.T == fe.T_patch + fe.T_slice


def test_build_frontend_factory():
    d = 64
    a = build_frontend("A", d, res=32, T=32, patch=4, depth=1)
    b = build_frontend("B", d, res=32, T=32, depth=1)
    c = build_frontend("C", d, res=32, T=32, depth=1)
    x = torch.rand(1, 3, 32, 32)
    ya, yb, yc = a(x), b(x), c(x)
    assert ya.tokens.shape[-1] == d and yb.tokens.shape[-1] == d and yc.tokens.shape[-1] == d
    assert yb.tokens.shape[1] == 32
    assert yc.tokens.shape[1] == yc.meta["T_patch"] + yc.meta["T_slice"]


if __name__ == "__main__":
    test_patch_token_count_and_T_grid()
    print("ok grid")
    test_A_shape_and_pool()
    print("ok A")
    test_B_multi_T_and_st_topk()
    print("ok B")
    test_C_budget_split()
    print("ok C")
    test_build_frontend_factory()
    print("ok factory")
    print("ALL FRONTEND TESTS PASSED")
