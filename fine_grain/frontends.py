"""Vision→LLM frontends A/B/C: unified visual tokens [B, T, d_llm].

A — pure patch (Uni-style grid embed + light MHSA stack)
B — pure slice tokens (ST + topk=2 strong baseline; T = slice_num, scalable)
C — coarse patch + fine slice concat (budget split reported)

All paths are torch-native (Conv2d / einsum / SDPA); no Python pixel loops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.models import (
    NS_A,
    NS_B,
    NS_C,
    NS_EPS,
    NS_STEPS_DEFAULT,
    AdaTempSlice,
    AttnPool,
    Block,
    SelfAttn,
    apply_slice_flags,
    coords,
    deslice_support_size,
)


def patch_token_count(res: int, patch: int) -> int:
    """Number of non-overlapping patch tokens for square images."""
    assert res % patch == 0, f"res={res} not divisible by patch={patch}"
    g = res // patch
    return g * g


def suggest_T_grid(res: int = 32, patch: int = 4) -> list[int]:
    """High T first (patch-matched), then compress: e.g. 64→32→16 for 32/p4."""
    tmax = patch_token_count(res, patch)
    grid = []
    t = tmax
    while t >= 8:
        grid.append(t)
        t = t // 2
    if 32 not in grid and 32 < tmax:
        grid.append(32)
    return sorted(set(grid), reverse=True)


@dataclass
class FrontendOut:
    tokens: torch.Tensor          # [B, T, d_llm]
    T: int
    meta: dict


class PatchFrontend(nn.Module):
    """A: fixed grid patchify → optional light stack → project to d_llm.

    If ``T`` < native patch count, vectorized adaptive pool on the token grid.
    If ``T`` > native, raise (caller should pick smaller patch or higher res).
    """

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        patch: int = 4,
        dim: int = 64,
        depth: int = 2,
        T: Optional[int] = None,
    ):
        super().__init__()
        assert res % patch == 0
        self.res, self.patch, self.dim = res, patch, dim
        self.grid = res // patch
        self.native_T = self.grid * self.grid
        self.T = int(T) if T is not None else self.native_T
        if self.T > self.native_T:
            raise ValueError(
                f"A: requested T={self.T} > native patch tokens {self.native_T}; "
                f"use smaller patch or larger res"
            )
        # Uni-style: patch embed (learned linear via conv) + pos
        self.stem = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.native_T, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, SelfAttn(dim)) for _ in range(depth)])
        self.proj = nn.Linear(dim, d_llm)

    def forward(self, img: torch.Tensor) -> FrontendOut:
        B, _, R, _ = img.shape
        assert R == self.res, f"expected res={self.res}, got {R}"
        x = self.stem(img)                                      # [B,dim,g,g]
        x = x.flatten(2).transpose(1, 2) + self.pos             # [B,Tn,dim]
        for b in self.blocks:
            x, _ = b(x)
        if self.T < self.native_T:
            # Vectorized length pool on token axis (works for non-square T)
            # [B, Tn, dim] -> [B, dim, Tn] -> adaptive_avg_pool1d -> [B, T, dim]
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.T).transpose(1, 2)
        tok = self.proj(x)
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={"kind": "A", "native_T": self.native_T, "patch": self.patch, "res": self.res},
        )


class SliceFrontend(nn.Module):
    """B: content-adaptive slice tokens; T = slice_num (scalable, not locked small).

    Strong baseline: local + mass + no_gumbel + ST + deslice_topk=2.
    Ambient C = heads*dim_head is raised when T is large so G<=C when possible.
    """

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        T: int = 64,
        dim: int = 64,
        depth: int = 2,
        deslice_topk: int = 2,
        stiefel: bool = True,
    ):
        super().__init__()
        self.res, self.T, self.dim = res, int(T), dim
        # Prefer G <= C: grow head dim so C >= T (cap for 4GB)
        heads = 4
        dim_head = max(16, (self.T + heads - 1) // heads)
        dim_head = min(dim_head, 64)  # cap
        C = heads * dim_head
        # If still G > C, keep T but log rank-deficient geometry in meta
        self.heads, self.dim_head, self.C = heads, dim_head, C
        # stem dim must match mixer dim = heads*dim_head for AdaTempSlice
        mix_dim = C
        self.mix_dim = mix_dim
        self.stem = nn.Linear(5, mix_dim)  # RGB + y + x
        self.local = nn.Sequential(
            nn.Conv2d(mix_dim, mix_dim, 3, padding=1, groups=mix_dim),
            nn.Conv2d(mix_dim, mix_dim, 1),
        )
        self.blocks = nn.ModuleList([
            Block(mix_dim, AdaTempSlice(mix_dim, heads=heads, dim_head=dim_head,
                                        slice_num=self.T, norm="mass"))
            for _ in range(depth)
        ])
        spec = dict(
            nog=True,
            stiefel_ns=bool(stiefel),
            deslice_topk=int(deslice_topk),
            ns_steps=NS_STEPS_DEFAULT,
        )
        # apply flags onto a fake module list via apply_slice_flags pattern
        class _Wrap(nn.Module):
            pass
        w = _Wrap()
        w.blocks = self.blocks
        apply_slice_flags(w, spec)
        self.proj = nn.Linear(mix_dim, d_llm)
        self._last_slot = {}

    def forward(self, img: torch.Tensor) -> FrontendOut:
        B, _, R, _ = img.shape
        assert R == self.res
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        x = self.stem(torch.cat([pts, p], -1))
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.local(g).flatten(2).transpose(1, 2)
        slices = None
        for b in self.blocks:
            x, slices = b(x)
        # slices: [B, G, mix_dim] from last block aux
        if slices is None:
            raise RuntimeError("slice mixer returned no tokens")
        tok = self.proj(slices)
        # effective-slot metrics from first mixer (vectorized)
        mix0 = self.blocks[0].mix
        slot = {}
        w = getattr(mix0, "last_w", None)
        w_write = getattr(mix0, "last_w_write", None)
        if w is not None:
            # PR_mass / r99 (same as collapse_stats, inline minimal)
            mass = w.sum(2)
            pp = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
            pr = float((1.0 / pp.pow(2).sum(-1)).mean())
            G = w.shape[-1]
            ps, _ = pp.sort(dim=-1, descending=True)
            cume = ps.cumsum(-1)
            hit = cume >= 0.99
            idx = hit.float().argmax(dim=-1)
            none = ~hit.any(dim=-1)
            r99 = (idx + 1).float()
            r99[none] = float(G)
            slot["PR_mass"] = pr
            slot["r99"] = float(r99.mean())
        if w_write is not None:
            slot["support"] = float(deslice_support_size(w_write))
        self._last_slot = slot
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "B",
                "T": self.T,
                "C": self.C,
                "G_le_C": self.T <= self.C,
                "deslice_topk": int(getattr(mix0, "deslice_topk", 0)),
                "stiefel_ns": bool(getattr(mix0, "stiefel_ns", False)),
                "slot": slot,
                "res": self.res,
            },
        )


class HybridFrontend(nn.Module):
    """C: coarse patch tokens + fine slice tokens; T = T_patch + T_slice."""

    def __init__(
        self,
        d_llm: int,
        res: int = 32,
        patch: int = 8,
        T_patch: Optional[int] = None,
        T_slice: int = 32,
        dim: int = 64,
        depth: int = 2,
        deslice_topk: int = 2,
    ):
        super().__init__()
        native = patch_token_count(res, patch)
        Tp = int(T_patch) if T_patch is not None else native
        self.patch_fe = PatchFrontend(
            d_llm, res=res, patch=patch, dim=dim, depth=depth, T=Tp,
        )
        self.slice_fe = SliceFrontend(
            d_llm, res=res, T=int(T_slice), dim=dim, depth=depth,
            deslice_topk=deslice_topk, stiefel=True,
        )
        self.T_patch = self.patch_fe.T
        self.T_slice = self.slice_fe.T
        self.T = self.T_patch + self.T_slice

    def forward(self, img: torch.Tensor) -> FrontendOut:
        pa = self.patch_fe(img)
        sl = self.slice_fe(img)
        tok = torch.cat([pa.tokens, sl.tokens], dim=1)
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "C",
                "T": tok.shape[1],
                "T_patch": pa.T,
                "T_slice": sl.T,
                "budget": f"{pa.T}+{sl.T}",
                "slice_slot": sl.meta.get("slot", {}),
                "res": self.patch_fe.res,
            },
        )


def build_frontend(
    kind: str,
    d_llm: int,
    res: int = 32,
    T: int = 64,
    patch: int = 4,
    T_patch: Optional[int] = None,
    T_slice: Optional[int] = None,
    dim: int = 64,
    depth: int = 2,
    deslice_topk: int = 2,
) -> nn.Module:
    kind = kind.upper()
    if kind == "A":
        return PatchFrontend(d_llm, res=res, patch=patch, dim=dim, depth=depth, T=T)
    if kind == "B":
        return SliceFrontend(
            d_llm, res=res, T=T, dim=dim, depth=depth, deslice_topk=deslice_topk,
        )
    if kind == "C":
        # default split: half budget each when T given
        ts = int(T_slice) if T_slice is not None else max(8, T // 2)
        # patch side: use coarser patch so native count ~ T - ts
        p = patch
        while patch_token_count(res, p) > max(8, T - ts) and p < res:
            p *= 2
        tp = int(T_patch) if T_patch is not None else min(patch_token_count(res, p), max(8, T - ts))
        return HybridFrontend(
            d_llm, res=res, patch=p, T_patch=tp, T_slice=ts,
            dim=dim, depth=depth, deslice_topk=deslice_topk,
        )
    raise ValueError(f"unknown frontend kind {kind}")
