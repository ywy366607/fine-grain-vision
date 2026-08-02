"""Tiny LLaVA-style + fine-grain synthetic batches (vectorized where possible)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from fine_grain.tasks import ANGLE_DEGS, make_angles, make_kinks, make_needle


def _caption_for_probe(kind: str, label: int) -> str:
    if kind == "needle":
        colors = ["red", "green", "blue", "yellow"]
        return f"The small square is {colors[int(label) % 4]}."
    if kind == "kinks":
        return f"The polyline has {int(label) + 5} corners."
    if kind == "angles":
        deg = ANGLE_DEGS[int(label) % len(ANGLE_DEGS)]
        return f"The angle is {deg} degrees."
    if kind == "caption":
        return "A synthetic scene with colored shapes."
    return "An image."


@torch.no_grad()
def make_llava_batch(
    rng: np.random.Generator,
    batch: int,
    res: int = 32,
    mix: Tuple[str, ...] = ("caption", "needle", "kinks"),
) -> Dict[str, object]:
    """On-the-fly mixed batch: images [B,3,H,W], texts list[str], probe tags.

    Fully generator-based; no disk I/O.
    """
    kinds = list(mix)
    imgs, texts, tags = [], [], []
    for i in range(batch):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        if kind == "needle":
            sizes = np.full(1, int(rng.choice([1, 2, 3, 4])))
            im, lab, _ = make_needle(rng, sizes, res=res)
            lab_i = int(lab[0].item())
        elif kind == "kinks":
            k = int(rng.choice([5, 6, 7, 8]))
            im, lab, _ = make_kinks(rng, np.array([k]), res=res, hard_frac=0.35, hard_tile=16)
            lab_i = int(lab[0].item())
        elif kind == "angles":
            deg = int(rng.choice(list(ANGLE_DEGS)))
            im, lab, _ = make_angles(rng, np.array([deg]), res=res)
            lab_i = int(lab[0].item())
        else:
            # cheap caption scene: reuse needle canvas
            sizes = np.full(1, int(rng.choice([2, 4, 6])))
            im, lab, _ = make_needle(rng, sizes, res=res)
            lab_i = int(lab[0].item())
            kind = "caption"
        imgs.append(im)
        texts.append(_caption_for_probe(kind, lab_i))
        tags.append({"probe": kind, "label": lab_i})
    img = torch.cat(imgs, 0)
    return {"image": img, "text": texts, "tags": tags}


def tokenize_captions(tokenizer, texts: List[str], max_length: int = 48):
    """Return input_ids, attention_mask on CPU long tensors."""
    out = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return out["input_ids"], out["attention_mask"]
