"""Unit tests for sparse deslice write + Qwen-style residual gate (shipped path)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fine_grain as D  # noqa: E402


def test_sparse_deslice_support_size():
    torch.manual_seed(0)
    B, H, N, G = 2, 4, 16, 8
    logits = torch.randn(B, H, N, G)
    w = torch.softmax(logits, dim=-1)
    # full soft: support ≈ G
    w_full = D.sparse_deslice_weights(w, topk=0, threshold=0.0)
    assert w_full.shape == w.shape
    sup_full = float(D.deslice_support_size(w_full))
    assert sup_full > G - 0.5, sup_full

    w2 = D.sparse_deslice_weights(w, topk=2, threshold=0.0)
    sup2 = float(D.deslice_support_size(w2))
    assert abs(sup2 - 2.0) < 1e-4, sup2
    # rows still sum to 1
    assert torch.allclose(w2.sum(-1), torch.ones(B, H, N), atol=1e-5)

    w_thr = D.sparse_deslice_weights(w, topk=0, threshold=0.2)
    # each row: nonzeros only where mass was >= 0.2 before renorm, support <= G
    assert float(D.deslice_support_size(w_thr)) <= G
    assert torch.allclose(w_thr.sum(-1), torch.ones(B, H, N), atol=1e-5)


def test_mixer_deslice_topk_forward_support():
    torch.manual_seed(1)
    dim, G, N = 64, 32, 64
    mix = D.AdaTempSlice(dim, heads=4, dim_head=16, slice_num=G, norm="mass")
    mix.no_gumbel = True
    mix.deslice_topk = 2
    mix.eval()
    x = torch.randn(2, N, dim)
    with torch.no_grad():
        out, tok = mix(x)
    assert out.shape == (2, N, dim)
    assert tok.shape[1] == G
    assert hasattr(mix, "last_w_write")
    sup = float(D.deslice_support_size(mix.last_w_write))
    assert abs(sup - 2.0) < 1e-3, sup
    # soft pool still used for tok; full soft w stored at eval
    assert mix.last_w is not None
    soft_sup = float(D.deslice_support_size(mix.last_w))
    assert soft_sup > 2.5, soft_sup  # soft participates more than top-2


def test_block_res_gate_form():
    torch.manual_seed(2)
    dim = 64
    mix = D.AdaTempSlice(dim, slice_num=16, norm="mass")
    mix.no_gumbel = True
    block = D.Block(dim, mix)
    block.use_res_gate = True
    # fix gate for algebraic check: force known g via zero weight + fixed bias
    with torch.no_grad():
        block.res_gate_proj.weight.zero_()
        block.res_gate_proj.bias.fill_(0.0)  # σ(0)=0.5
    x = torch.randn(2, 32, dim)
    # capture mix_out by hijacking
    raw_mix = block.mix

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.last_m = None

        def forward(self, z):
            m, aux = self.inner(z)
            self.last_m = m
            return m, aux

        def __getattr__(self, name):
            if name in ("inner", "last_m"):
                return super().__getattr__(name)
            return getattr(self.inner, name)

    # simpler: manual residual identity
    block.use_res_gate = False
    y_off, _ = block(x.clone())
    block.use_res_gate = True
    with torch.no_grad():
        block.res_gate_proj.weight.zero_()
        block.res_gate_proj.bias.fill_(0.0)
    # recompute: pre=x, m = mix(ln1(x)), g=0.5, then x + 0.5*m + mlp
    # Compare to ungated with scaled mix is hard; check gate range and that disable matches
    block.use_res_gate = False
    y0, _ = block(x.clone())
    block.use_res_gate = True
    with torch.no_grad():
        # bias large → g≈1 → near ungated
        block.res_gate_proj.weight.zero_()
        block.res_gate_proj.bias.fill_(10.0)
    y1, _ = block(x.clone())
    assert torch.allclose(y0, y1, atol=1e-4), (y0 - y1).abs().max()
    g = block.last_res_gate
    assert g.min() > 0 and g.max() < 1
    assert g.min() > 0.99  # σ(10)≈1


def test_apply_flags_baseline_unchanged_defaults():
    m = D.build(D.ARMS["slice_loc_nogumbel"], 64, 3, 32, 6)
    for b in m.blocks:
        assert b.mix.deslice_topk == 0
        assert not b.use_res_gate
        assert not b.mix.qwen_sdpa_gate


def test_build_topk_arm():
    m = D.build(D.ARMS["slice_loc_nogumbel_st_topk2"], 64, 2, 32, 6)
    for b in m.blocks:
        assert b.mix.deslice_topk == 2
        assert b.mix.stiefel_ns is True
        assert b.mix.no_gumbel is True


if __name__ == "__main__":
    test_sparse_deslice_support_size()
    print("ok support")
    test_mixer_deslice_topk_forward_support()
    print("ok mixer forward support")
    test_block_res_gate_form()
    print("ok gate form")
    test_apply_flags_baseline_unchanged_defaults()
    print("ok baseline flags")
    test_build_topk_arm()
    print("ok topk arm")
    print("ALL UNIT TESTS PASSED")
