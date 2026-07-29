"""Unit tests for sparse deslice write + Qwen-style gates (shipped path).

Drives real fine_grain.models entry points — no reimplementation of deslice math.
"""
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
    w_full = D.sparse_deslice_weights(w, topk=0, threshold=0.0)
    assert w_full.shape == w.shape
    sup_full = float(D.deslice_support_size(w_full))
    assert sup_full > G - 0.5, sup_full

    w2 = D.sparse_deslice_weights(w, topk=2, threshold=0.0)
    sup2 = float(D.deslice_support_size(w2))
    assert abs(sup2 - 2.0) < 1e-4, sup2
    assert torch.allclose(w2.sum(-1), torch.ones(B, H, N), atol=1e-5)

    w_thr = D.sparse_deslice_weights(w, topk=0, threshold=0.2)
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
    assert soft_sup > 2.5, soft_sup


def test_soft_deslice_uses_all_slices():
    """Disabled top-k must leave full soft support (~G)."""
    torch.manual_seed(3)
    mix = D.AdaTempSlice(64, slice_num=16, norm="mass")
    mix.no_gumbel = True
    mix.deslice_topk = 0
    mix.eval()
    with torch.no_grad():
        mix(torch.randn(1, 32, 64))
    sup = float(D.deslice_support_size(mix.last_w_write))
    assert sup > 14.0, sup  # nearly all 16 slices participate


def test_block_res_gate_form():
    """x' = x + σ(W x_pre) ⊙ mix_out  (Qwen residual-stream branch gate)."""
    torch.manual_seed(2)
    dim = 64
    mix = D.AdaTempSlice(dim, slice_num=16, norm="mass")
    mix.no_gumbel = True
    block = D.Block(dim, mix)
    x = torch.randn(2, 32, dim)

    block.use_res_gate = False
    y0, _ = block(x.clone())
    block.use_res_gate = True
    with torch.no_grad():
        # σ(10)≈1 → near ungated residual mix path (MLP still same)
        block.res_gate_proj.weight.zero_()
        block.res_gate_proj.bias.fill_(10.0)
    y1, _ = block(x.clone())
    assert torch.allclose(y0, y1, atol=1e-4), (y0 - y1).abs().max()
    g = block.last_res_gate
    assert g.min() > 0.99  # σ(10)≈1
    assert g.max() < 1.0 + 1e-5

    # algebraic: force g=0.5, compare to half-mix residual + mlp
    with torch.no_grad():
        block.res_gate_proj.weight.zero_()
        block.res_gate_proj.bias.fill_(0.0)  # σ(0)=0.5
    # manual one-step residual mix with g=0.5
    pre = x.clone()
    m, _ = block.mix(block.ln1(pre))
    x_half = pre + 0.5 * m
    y_half = x_half + block.mlp(block.ln2(x_half))
    y_g, _ = block(x.clone())
    assert torch.allclose(y_g, y_half, atol=1e-4), (y_g - y_half).abs().max()


def test_qwen_sdpa_gate_multiplies_att_not_qk():
    """SDPA output gate (arXiv:2505.06708) is after attention on slice tokens."""
    torch.manual_seed(4)
    mix = D.AdaTempSlice(64, heads=4, dim_head=16, slice_num=8, norm="mass")
    mix.no_gumbel = True
    mix.qwen_sdpa_gate = True
    mix.eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        mix.to_out.bias.zero_()  # isolate gate effect from output bias
        mix.sdpa_gate_proj.weight.zero_()
        mix.sdpa_gate_proj.bias.fill_(-20.0)  # σ(-20)≈0 → att killed
        out0, _ = mix(x)
        mix.sdpa_gate_proj.bias.fill_(20.0)   # σ(20)≈1
        out1, _ = mix(x)
    assert out0.abs().mean() < 1e-4, float(out0.abs().mean())
    assert out1.abs().mean() > out0.abs().mean() + 1e-3
    g = mix.last_sdpa_gate
    assert g is not None and g.shape[-1] == 1


def test_apply_flags_baseline_unchanged_defaults():
    m = D.build(D.ARMS["slice_loc_nogumbel"], 64, 3, 32, 6)
    for b in m.blocks:
        assert b.mix.deslice_topk == 0
        assert not b.use_res_gate
        assert not b.mix.qwen_sdpa_gate
        assert int(getattr(b, "recur_T", 1)) == 1


def test_build_topk_arm():
    m = D.build(D.ARMS["slice_loc_nogumbel_st_topk2"], 64, 2, 32, 6)
    for b in m.blocks:
        assert b.mix.deslice_topk == 2
        assert b.mix.stiefel_ns is True
        assert b.mix.no_gumbel is True


def test_recur_T_shared_weights_active():
    """Fallback multi-pass: recur_T>1 runs shared mix T times (not primary scatter fix)."""
    torch.manual_seed(5)
    m = D.build(D.ARMS["slice_loc_nogumbel_recur2"], 64, 1, 16, 2)
    assert m.blocks[0].recur_T == 2
    m.eval()
    img = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        # monkey-count mix calls
        calls = {"n": 0}
        raw = m.blocks[0].mix.forward

        def counted(x):
            calls["n"] += 1
            return raw(x)

        m.blocks[0].mix.forward = counted
        _ = m(img)
    assert calls["n"] == 2, calls
    assert m.blocks[0].last_recur_T == 2


if __name__ == "__main__":
    test_sparse_deslice_support_size()
    print("ok support")
    test_mixer_deslice_topk_forward_support()
    print("ok mixer forward support")
    test_soft_deslice_uses_all_slices()
    print("ok soft full support")
    test_block_res_gate_form()
    print("ok gate form")
    test_qwen_sdpa_gate_multiplies_att_not_qk()
    print("ok qwen sdpa gate")
    test_apply_flags_baseline_unchanged_defaults()
    print("ok baseline flags")
    test_build_topk_arm()
    print("ok topk arm")
    test_recur_T_shared_weights_active()
    print("ok recur fallback")
    print("ALL UNIT TESTS PASSED")
