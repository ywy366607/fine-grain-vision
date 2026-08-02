"""Unit test: probe_token_acc alignment must match causal LM teacher-forcing."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_vlm_frontends import align_text_token_preds  # noqa: E402


def test_align_text_token_preds_teacher_forced_identity():
    """Build logits that peak at the true next text id for every position.

    Sequence: T visual + L text. logits[b, T-1+j, ids[b,j]] = large → pred == ids.
    """
    B, T, L, V = 2, 4, 5, 32
    ids = torch.tensor([
        [3, 7, 11, 2, 9],
        [1, 4, 4, 8, 0],
    ], dtype=torch.long)
    mask = torch.tensor([
        [1, 1, 1, 1, 0],  # last pad
        [1, 1, 1, 0, 0],
    ], dtype=torch.long)
    logits = torch.zeros(B, T + L, V)
    for b in range(B):
        for j in range(L):
            # position that predicts ids[b,j] is T-1+j
            logits[b, T - 1 + j, ids[b, j]] = 10.0

    pred, tgt, valid = align_text_token_preds(logits, T, ids, mask)
    assert pred.shape == (B, L)
    assert torch.equal(tgt, ids)
    # all non-pad positions must match
    assert (pred[valid] == ids[valid]).all(), (pred, ids, valid)
    # accuracy 1.0 on valid
    acc = (pred[valid] == ids[valid]).float().mean().item()
    assert abs(acc - 1.0) < 1e-6

    # wrong slice (old bug L-1 length) must NOT be used: length check
    bad = logits[:, T - 1 : T - 1 + L - 1].argmax(-1)
    assert bad.shape[1] == L - 1
    assert bad.shape != ids.shape


def test_align_detects_shift_bug():
    """If preds are shifted by +1 vs targets, valid accuracy collapses."""
    B, T, L, V = 1, 3, 4, 16
    ids = torch.tensor([[5, 6, 7, 8]])
    mask = torch.ones(1, 4, dtype=torch.long)
    logits = torch.zeros(B, T + L, V)
    # deliberately put correct mass at T+j (one step too late for ids[j])
    for j in range(L):
        logits[0, T + j, ids[0, j]] = 10.0  # would predict ids[j] for position after T+j
    pred, tgt, valid = align_text_token_preds(logits, T, ids, mask)
    # correct positions T-1+j have zero peak → argmax not ids
    acc = (pred[valid] == tgt[valid]).float().mean().item()
    assert acc < 0.5, acc


if __name__ == "__main__":
    test_align_text_token_preds_teacher_forced_identity()
    print("ok identity")
    test_align_detects_shift_bug()
    print("ok shift detect")
    print("ALL PROBE ALIGNMENT TESTS PASSED")
