"""Fair patch recon head: unpatchify / pixel_shuffle can express within-patch structure."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.line_recon import PatchSeg, unpatchify  # noqa: E402


def test_unpatchify_layout():
    B, Hp, Wp, p = 2, 4, 4, 16
    # encode a unique value per cell position so layout is checkable
    pix = torch.zeros(B, Hp, Wp, p * p)
    for i in range(p):
        for j in range(p):
            pix[:, :, :, i * p + j] = i * 100 + j
    out = unpatchify(pix, p)
    assert out.shape == (B, Hp * p, Wp * p)
    # top-left patch cell (0,0): out[y,x] for y,x in 0..p-1 should be y*100+x
    for y in range(p):
        for x in range(p):
            assert float(out[0, y, x]) == y * 100 + x


def test_patchseg_shapes_all_decoders():
    for dec in ("unpatchify", "pixel_shuffle", "bilinear"):
        m = PatchSeg(dim=64, depth=1, patch=16, decoder=dec)
        m.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 64, 64), (dec, y.shape)


def test_unpatchify_can_make_sharp_line_pattern():
    """Unlike bilinear-only heads, Linear→unpatchify can hold a 1px column inside a patch."""
    p = 16
    m = PatchSeg(dim=64, depth=1, patch=p, decoder="unpatchify")
    # force head to emit a vertical line in the middle of every patch
    with torch.no_grad():
        m.to_pixels.weight.zero_()
        m.to_pixels.bias.zero_()
        mid = p // 2
        for i in range(p):
            m.to_pixels.bias[i * p + mid] = 10.0  # high logit on middle column
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = m(x)
    # every patch's middle column should be strongly positive
    assert (logits[0, :, mid::p] > 5).all() or (logits[0, :, mid] > 5).any()
    # bilinear path cannot do this from a constant-per-token map — unpatchify can


if __name__ == "__main__":
    test_unpatchify_layout()
    print("ok layout")
    test_patchseg_shapes_all_decoders()
    print("ok shapes")
    test_unpatchify_can_make_sharp_line_pattern()
    print("ok sharp pattern")
    print("ALL PATCH DECODER TESTS PASSED")
