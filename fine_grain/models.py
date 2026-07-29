"""Slice and patch encoders for fine-grained vision experiments.

AdaTempSlice is ported from Transolver++ (thuml/Transolver_plus) with optional
extensions: no-Gumbel assignment, stride-1 local DW conv, Stiefel Newton–Schulz
directions, sparse deslice write, and Qwen-style residual/SDPA gates.

Soft-deslice noise (primary)
----------------------------
Transolver's write path is soft MoE scatter: every point receives a mixture of
all G slice messages. That leaks energy off the thin structure into background
pixels (line-recon FP). **Primary fix** is sparse/thresholded *write* weights
(``sparse_deslice_weights``) while mass-norm soft *read/pool* stays soft for
small-target amplitude and gradients.

Qwen gated attention (secondary, correct placement)
---------------------------------------------------
Qiu et al., arXiv:2505.06708 (NeurIPS 2025 Best Paper; used in Qwen3-Next):
head-specific ``O ← σ(W x) ⊙ SDPA(Q,K,V)`` — gate is **after** SDPA, not on
QK softmax and not only on the task head. Mapped here as:
  (1) ``AdaTempSlice.qwen_sdpa_gate``: σ-gate multiplies **slice-token** SDPA
      output before deslice (query-side features = mass-normed tokens);
  (2) ``Block.use_res_gate``: residual-stream form ``x + σ(W x_pre) ⊙ mix_out``
      (gate multiplies the branch entering residual add).

Recurrence (fallback only)
--------------------------
Shared-weight multi-pass mix (Universal Transformer / iterative refinement)
is **not** the primary scatter fix. Enable ``Block.recur_T > 1`` only after
sparse write is measured; default T=1 (single pass).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.tasks import N_CLASSES

# Align with torch.optim.Muon / torch.optim._muon (PyTorch 2.9+):
#   DEFAULT_A,B,C = 3.4445, -4.7750, 2.0315 ; DEFAULT_NS_STEPS = 5 ; EPS = 1e-7
#   source: torch/optim/_muon.py :: _zeropower_via_newtonschulz
#   (Keller Jordan Muon post; quintic NS maximizing slope at 0)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_STEPS_DEFAULT = 5
NS_EPS = 1e-7


def sparse_deslice_weights(w, topk=0, threshold=0.0):
    """Build write/scatter weights for deslice; keeps soft *pool* weights untouched.

    Primary fix for soft-scatter leakage (Transolver soft MoE write path):
    - topk>0: keep only top-k slice masses per point (per head), renormalize.
    - threshold>0: zero masses below threshold, renormalize.
    - both 0: full soft scatter (baseline Transolver++).

    w: [B,H,N,G] assignment after softmax. Returns same shape.
    """
    topk = int(topk or 0)
    thr = float(threshold or 0.0)
    G = w.shape[-1]
    if topk <= 0 and thr <= 0:
        return w
    w_sp = w
    if topk > 0 and topk < G:
        vals, idx = torch.topk(w, k=topk, dim=-1)
        w_sp = torch.zeros_like(w)
        w_sp.scatter_(-1, idx, vals)
    if thr > 0:
        w_sp = torch.where(w_sp >= thr, w_sp, torch.zeros_like(w_sp))
    # If a point lost all mass, fall back to original soft row (avoid NaN)
    row = w_sp.sum(dim=-1, keepdim=True)
    empty = row < 1e-8
    if empty.any():
        w_sp = torch.where(empty.expand_as(w_sp), w, w_sp)
        row = w_sp.sum(dim=-1, keepdim=True)
    return w_sp / row.clamp_min(1e-8)


def deslice_support_size(w_write):
    """Mean number of non-negligible slice weights per (B,H,N) after write sparsification."""
    return (w_write > 1e-6).float().sum(dim=-1).mean()


def newton_schulz(X, steps=NS_STEPS_DEFAULT, coefficients=(NS_A, NS_B, NS_C), eps=NS_EPS):
    """Newton–Schulz zeropower / polar orthogonalization, batched.

    Matches torch.optim._muon._zeropower_via_newtonschulz (same a,b,c,steps,eps and
    the same quintic recurrence), extended to batch dims [..., m, n].

    When m >= n: orthonormal columns in R^m (Stiefel St(m,n)).
    When n > m: dual/wide case (same as Muon: transpose, iterate, transpose back).
    """
    a, b, c = coefficients
    # Muon ref uses bfloat16 for the iterate; keep compute dtype of X for amp-friendliness
    # but run NS in float32 for stability under autocast.
    dtype_in = X.dtype
    X = X.float()
    # spectral norm <= 1 (Frobenius over the last two dims, per batch row)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    # Muon: if size(0) > size(1) work on X.T so gram is the smaller face
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-1, -2)
    for _ in range(int(steps)):
        # gram = X @ X^T  (Muon)  — here X is [..., m', n'] with m' <= n'
        gram = X @ X.transpose(-1, -2)
        # gram_update = b * gram + c * gram @ gram
        gram_update = b * gram + c * (gram @ gram)
        # X <- a * X + gram_update @ X
        X = a * X + gram_update @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(dtype=dtype_in)


class AdaTempSlice(nn.Module):
    """Physics-Attention with eidetic states, ported from thuml/Transolver_plus
    (models/Transolver_plus.py :: Physics_Attention_1D_Eidetic).

    Deviations (each changes numbers; default keeps reference-like soft path):
    (1) no distributed all_reduce; (2) Gumbel off at eval; (3) optional Stiefel NS
    on slice directions after mass-norm; (4) **sparse deslice write** — mass *pool*
    stays soft for gradients/small-target amplitude, but scatter/deslice may use
    top-k / thresholded weights to cut soft-scatter spatial leakage (primary noise
    on line-recon; Transolver soft-MoE write tax); (5) optional **Qwen gated
    attention** (Qiu et al. arXiv:2505.06708, NeurIPS 2025 Best Paper; used in
    Qwen3-Next): head-specific sigmoid gate **after SDPA**,
    ``G=σ(x W_θ)``, ``O ← G ⊙ SDPA(Q,K,V)`` — not on QK softmax, not task head.
    Residual-stream post-mix gate (block-level) is separate — see ``Block``.
    """

    def __init__(self, dim, heads=4, dim_head=16, slice_num=32, norm="mass"):
        super().__init__()
        assert norm in ("mass", "const", "none")
        self.h, self.dh, self.g, self.norm = heads, dim_head, slice_num, norm
        inner = heads * dim_head
        self.in_project_x = nn.Linear(dim, inner)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)      # as in the reference
        self.proj_temperature = nn.Sequential(                 # Ada-Temp
            nn.Linear(dim_head, slice_num), nn.GELU(),
            nn.Linear(slice_num, 1), nn.GELU())
        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Linear(inner, dim)
        # optional: Newton–Schulz Stiefel/polar on directions (see forward)
        self.stiefel_ns = False
        self.ns_steps = NS_STEPS_DEFAULT
        self.ns_coefficients = (NS_A, NS_B, NS_C)
        self.ns_eps = NS_EPS
        # sparse WRITE path (pool/read still uses soft w)
        self.deslice_topk = 0          # 0 = full soft scatter
        self.deslice_threshold = 0.0   # 0 = no threshold
        # Qwen gated attention: head-specific gate after SDPA (default off)
        self.qwen_sdpa_gate = False
        # gate from residual-stream point features after in_project: [B,N,inner] -> [B,H,N,1]
        self.sdpa_gate_proj = nn.Linear(dim_head, 1)

    def forward(self, x):                                      # x: [B,N,C]
        B, N, _ = x.shape
        xm = self.in_project_x(x).reshape(B, N, self.h, self.dh).permute(0, 2, 1, 3)

        temp = torch.clamp(self.proj_temperature(xm) + self.bias, min=0.01)
        logits = self.in_project_slice(xm)                     # [B,H,N,G]
        # no_gumbel: diagnostic switch (scripts/_diag_slice_capacity.py). The reference
        # samples Gumbel noise on every assignment; if the glyph failure is an
        # OPTIMIZATION failure rather than an expressivity one, that noise is a prime
        # suspect. Absent the attribute this is exactly the reference behaviour.
        if self.training and not getattr(self, "no_gumbel", False):
            u = torch.rand_like(logits)
            logits = logits - torch.log(-torch.log(u + 1e-8) + 1e-8)
        w = F.softmax(logits / temp, dim=-1)
        mass = w.sum(2) + 1e-5                                 # [B,H,G] assignment mass
        # --- READ / pool: always soft w (+ mass-norm) for small-target amplitude ---
        tok = torch.einsum("bhnc,bhng->bhgc", xm, w)
        if self.norm == "mass":
            # The whole claim: dividing by assignment mass makes a slice owned by one
            # pixel carry that pixel at full amplitude, independent of how few pixels
            # it is. This is the reference Transolver++ behaviour.
            tok = tok / mass.unsqueeze(-1)
        elif self.norm == "const":
            # Clean control (v2): a CONSTANT denominator keeps activation scale sane but
            # removes the content-adaptive part. Isolates "size invariance" from "scale
            # control", which slice_sum conflated.
            tok = tok / (N / self.g)

        # Stiefel / polar on DIRECTIONS only; scale kept from mass-normalized tok.
        # Concat heads -> X[B,C,G], C=h*dh: when C>=G this is classical St(C,G)
        # (orthonormal slice axes in full channel space). Mass-norm already set
        # content amplitude; we store per-slice ||tok|| and re-apply after NS so
        # Newton–Schulz only moves directions (Muon-style spectral multi-axis).
        if getattr(self, "stiefel_ns", False):
            C = self.h * self.dh
            X = tok.permute(0, 1, 3, 2).reshape(B, C, self.g)       # [B,C,G]
            mag = X.norm(dim=1, keepdim=True).clamp_min(1e-6)      # [B,1,G] scale
            U = newton_schulz(
                X,
                steps=getattr(self, "ns_steps", NS_STEPS_DEFAULT),
                coefficients=getattr(self, "ns_coefficients", (NS_A, NS_B, NS_C)),
                eps=getattr(self, "ns_eps", NS_EPS),
            )
            X_st = U * mag                                         # direction@Stiefel × mass scale
            tok = X_st.reshape(B, self.h, self.dh, self.g).permute(0, 1, 3, 2)
            self.last_mass = mass.detach()

        att = F.scaled_dot_product_attention(self.to_q(tok), self.to_k(tok), self.to_v(tok))
        # Qwen gated attention (arXiv:2505.06708): head-specific σ-gate *after* SDPA
        # on the attention *output* (slice tokens), not on QK softmax.
        # Query-side features = mass-normalized slice tokens `tok` [B,H,G,Dh]
        # (Transolver's "query positions" are the G slices). Form:
        #   G = σ(tok W_θ) ∈ (0,1)^{B,H,G,1};  att ← G ⊙ att
        if getattr(self, "qwen_sdpa_gate", False):
            g_tok = torch.sigmoid(self.sdpa_gate_proj(tok))        # [B,H,G,1]
            att = att * g_tok
            self.last_sdpa_gate = g_tok.detach()
        else:
            self.last_sdpa_gate = None

        # --- WRITE / deslice: optional sparse scatter (PRIMARY soft-scatter fix) ---
        # Soft pool/read above is unchanged; only the write path may sparsify.
        w_write = sparse_deslice_weights(
            w,
            topk=getattr(self, "deslice_topk", 0),
            threshold=getattr(self, "deslice_threshold", 0.0),
        )
        out = torch.einsum("bhgc,bhng->bhnc", att, w_write)        # deslice
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.h * self.dh)

        # probes
        self.last_temp = temp.detach()
        self.last_w_write = w_write.detach()
        if not self.training:
            self.last_w = w.detach()
            self.last_tok = tok.detach()
            self.last_mass = mass.detach()
            self.last_support = deslice_support_size(w_write).detach()

        return self.to_out(out), tok.permute(0, 2, 1, 3).reshape(B, self.g, self.h * self.dh)


class SelfAttn(nn.Module):
    """Plain MHSA over the patch tokens -- what a ViT block actually does."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.out(a.transpose(1, 2).reshape(B, N, C)), x


class Block(nn.Module):
    """Pre-norm block. Residual write of the mixer branch may be gated:

    ``x ← x + σ(W_g x_pre) ⊙ mix_out`` when ``res_gate`` is enabled.

    This is the residual-stream analogue of Qwen post-attn write control
    (arXiv:2505.06708): gate multiplies the *branch* entering residual add,
    from residual-stream features — not QK scores, not the task head.
    Slice-token SDPA output gates live inside ``AdaTempSlice.qwen_sdpa_gate``.

    Optional **fallback** refinement: ``recur_T > 1`` reuses the same mixer
    weights T times (Universal Transformer / iterative refinement style).
    Default T=1. Prefer sparse deslice for soft-scatter first; recurrence is
    not the primary noise fix.
    """

    def __init__(self, dim, mixer, mlp_ratio=2):
        super().__init__()
        self.ln1, self.ln2, self.mix = nn.LayerNorm(dim), nn.LayerNorm(dim), mixer
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))
        # optional residual-stream gate (default off — prior arms unchanged)
        self.use_res_gate = False
        self.res_gate_proj = nn.Linear(dim, dim)
        # start near-open so early training ≈ ungated residual
        nn.init.zeros_(self.res_gate_proj.weight)
        nn.init.constant_(self.res_gate_proj.bias, 2.0)
        # fallback shared-weight multi-pass (default single pass)
        self.recur_T = 1

    def _mix_residual(self, x):
        """One residual mix step: x + [gate ⊙] mix(LN(x))."""
        pre = x
        m, aux = self.mix(self.ln1(x))
        if getattr(self, "use_res_gate", False):
            g = torch.sigmoid(self.res_gate_proj(pre))
            self.last_res_gate = g.detach()
            x = x + g * m
        else:
            x = x + m                       # point stream survives as a residual
        return x, aux

    def forward(self, x):
        T = max(1, int(getattr(self, "recur_T", 1) or 1))
        aux = None
        for _ in range(T):
            x, aux = self._mix_residual(x)
        self.last_recur_T = T
        return x + self.mlp(self.ln2(x)), aux


class AttnPool(nn.Module):
    """Learned-query attention pool. Identical module in every arm, so the readout is
    never the confound. Deliberately NOT a mean pool -- a mean pool would re-introduce
    exactly the area-proportional dilution the experiment is about."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.kv = nn.Linear(dim, dim * 2)

    def forward(self, tok):
        k, v = self.kv(tok).chunk(2, -1)
        a = torch.softmax(k @ self.q / k.shape[-1] ** 0.5, dim=-1)
        return (a.unsqueeze(-1) * v).sum(1)


# --------------------------------------------------------------------------- models

def coords(R, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, R, device=device),
                          torch.linspace(-1, 1, R, device=device), indexing="ij")
    return torch.stack([y, x], -1).reshape(1, R * R, 2)


class SliceNet(nn.Module):
    def __init__(self, dim=64, depth=3, slice_num=32, norm="mass", n_cls=N_CLASSES,
                 readout="slices", local=False, n_freq=0):
        super().__init__()
        assert readout in ("slices", "points")
        self.readout, self.n_freq = readout, n_freq
        # n_freq>0: Fourier position features instead of raw (y,x). Averaging raw
        # coordinates over a slice keeps only the CENTROID (a first moment); averaging
        # sin/cos(w.p) keeps samples of the slice's spatial CHARACTERISTIC FUNCTION, i.e.
        # its whole distribution. Proposed 2026-07-29, then dropped on the Step-0 result
        # ("aggregation only costs .018") -- WRONGLY: that .018 is conditional on THIS
        # stem, and this arm changes the stem, so the ratio does not transfer. Restored
        # and run at the user's prompting; the drop was an inference, never a measurement.
        self.stem = nn.Linear(3 + (4 * n_freq if n_freq else 2), dim)
        # STRIDE-1 local mixing (added 2026-07-29 after the Step-0 probe). The probe
        # decomposed the glyph gap as: aggregation costs only 0.018 (slice tokens 0.398
        # vs point stream 0.416) while patch tokens sit at 0.800 -- so the shape gap is
        # upstream of aggregation, and the slice stack simply has NO operator that looks
        # at a neighbourhood (per-point Linear stem + globally content-addressed mixing).
        # patchify builds shape because it IS a local filter; its stride is a separate
        # thing and is what buries a 1px object. Slice attention is O(NM), so unlike a
        # ViT we can afford locality WITHOUT the stride. Depthwise-separable (4.7k par)
        # rather than a full 3x3 (36.9k) so a win cannot be bought with parameters.
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)) if local else None
        self.blocks = nn.ModuleList([
            Block(dim, AdaTempSlice(dim, slice_num=slice_num, norm=norm))
            for _ in range(depth)])
        self.pool, self.head = AttnPool(dim), nn.Linear(dim, n_cls)

    def forward(self, img):
        B, _, R, _ = img.shape
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device)
        if self.n_freq:
            f = (2.0 ** torch.arange(self.n_freq, device=img.device)) * math.pi
            a = p[..., None] * f                            # [1,N,2,F]
            p = torch.cat([a.sin(), a.cos()], -1).flatten(-2)
        x = self.stem(torch.cat([pts, p.expand(B, -1, -1)], -1))
        if self.local is not None:                          # residual, stride 1, no pooling
            g = x.transpose(1, 2).reshape(B, -1, R, R)
            x = x + self.local(g).flatten(2).transpose(1, 2)
        slices = None
        for b in self.blocks:
            x, slices = b(x)
        # readout="slices" (v1 default): deliberately conservative -- 32 tokens, FEWER than
        #   patch4's 64, so a WIN cannot be explained by giving this arm more to look at.
        # readout="points" (added 2026-07-29 after glyph): the FAITHFUL analogue of the
        #   reference, whose output head reads the POINT stream (Model.forward ends in
        #   mlp2(ln_3(fx)) with fx per-point). The conservative choice is a handicap that
        #   could by itself explain the glyph failure, since arrangement survives in the
        #   point residual and slice-token-only readout throws it away.
        return self.head(self.pool(slices if self.readout == "slices" else x))


class PatchNet(nn.Module):
    def __init__(self, dim=64, depth=3, patch=4, n_cls=N_CLASSES):
        super().__init__()
        self.patch = patch
        self.stem = nn.Conv2d(3, dim - 2, patch, patch)     # exactly ViT patchify
        self.blocks = nn.ModuleList([Block(dim, SelfAttn(dim)) for _ in range(depth)])
        self.pool, self.head = AttnPool(dim), nn.Linear(dim, n_cls)

    def forward(self, img):
        B = img.shape[0]
        f = self.stem(img)
        Rp = f.shape[-1]
        x = f.reshape(B, -1, Rp * Rp).transpose(1, 2)
        x = torch.cat([x, coords(Rp, img.device).expand(B, -1, -1)], -1)   # same pos info
        for b in self.blocks:
            x, _ = b(x)
        return self.head(self.pool(x))


ARMS = {
    "patch4":      dict(kind="patch", patch=4, mult=1),
    "patch8":      dict(kind="patch", patch=8, mult=1),
    "patch16":     dict(kind="patch", patch=16, mult=1),  # standard ViT-scale patch
    "patch4_hi":   dict(kind="patch", patch=4, mult=2),
    "slice":       dict(kind="slice", norm="mass",  mult=1),
    "slice_const": dict(kind="slice", norm="const", mult=1),   # v2 clean control
    "slice_sum":   dict(kind="slice", norm="none",  mult=1),   # v1 dirty control
    # Oracle (2026-07-29): Gumbel on assignment is the prime optimisation bottleneck on
    # glyph memorisation (slice 0.812@1200 still climbing; slice_nogumbel 1.000@900).
    # This arm is the generalisation test of that diagnosis -- not a new mechanism story.
    "slice_nogumbel": dict(kind="slice", norm="mass", mult=1, nog=True),
    # v2b: faithful readout (the reference reads the POINT stream, not the slice tokens).
    "slice_pt":    dict(kind="slice", norm="mass",  mult=1, readout="points"),
    "slice_pt_nogumbel": dict(kind="slice", norm="mass", mult=1, readout="points", nog=True),
    # v3: stride-1 locality + slice aggregation. The design under test.
    "slice_loc":   dict(kind="slice", norm="mass",  mult=1, local=True),
    "slice_loc_pt": dict(kind="slice", norm="mass", mult=1, local=True, readout="points"),
    "slice_loc_nogumbel": dict(kind="slice", norm="mass", mult=1, local=True, nog=True),
    "slice_loc_pt_nogumbel": dict(kind="slice", norm="mass", mult=1, local=True,
                                  readout="points", nog=True),
    # Stiefel directions via Newton–Schulz after mass-norm (scale still from mass path)
    "slice_loc_nogumbel_st": dict(kind="slice", norm="mass", mult=1, local=True,
                                  nog=True, stiefel_ns=True),
    # Primary soft-scatter fix: top-k deslice write (soft mass read kept)
    "slice_loc_nogumbel_topk2": dict(kind="slice", norm="mass", mult=1, local=True,
                                     nog=True, deslice_topk=2),
    "slice_loc_nogumbel_st_topk2": dict(kind="slice", norm="mass", mult=1, local=True,
                                        nog=True, stiefel_ns=True, deslice_topk=2),
    # Qwen-style residual-stream post-mix gate (+ optional sparse write)
    "slice_loc_nogumbel_gate": dict(kind="slice", norm="mass", mult=1, local=True,
                                    nog=True, res_gate=True),
    "slice_loc_nogumbel_st_topk2_gate": dict(
        kind="slice", norm="mass", mult=1, local=True, nog=True,
        stiefel_ns=True, deslice_topk=2, res_gate=True, qwen_sdpa_gate=True),
    # Fallback only: shared-weight multi-pass mix (not the primary scatter fix)
    "slice_loc_nogumbel_recur2": dict(
        kind="slice", norm="mass", mult=1, local=True, nog=True, recur_T=2),
    "slice_loc_nogumbel_topk2_recur2": dict(
        kind="slice", norm="mass", mult=1, local=True, nog=True,
        deslice_topk=2, recur_T=2),
    # A1, restored 2026-07-29: Fourier position basis in the stem (never run before).
    "slice_four":    dict(kind="slice", norm="mass", mult=1, n_freq=4),
    "slice_four_pt": dict(kind="slice", norm="mass", mult=1, n_freq=4, readout="points"),
    "slice_four_loc": dict(kind="slice", norm="mass", mult=1, n_freq=4, local=True),
    # Fourier + no Gumbel: only meaningful once oracle showed Gumbel is an optim poison;
    # bare slice_four on a Gumbel baseline would re-test a known failure mode.
    "slice_four_nogumbel": dict(kind="slice", norm="mass", mult=1, n_freq=4, nog=True),
}


def apply_slice_flags(model, spec):
    """Apply optional mixer/block flags from an ARMS-style dict (idempotent)."""
    for b in model.blocks:
        mix = b.mix
        if spec.get("nog"):
            mix.no_gumbel = True
        if spec.get("stiefel_ns"):
            mix.stiefel_ns = True
            mix.ns_steps = int(spec.get("ns_steps", NS_STEPS_DEFAULT))
            mix.ns_coefficients = (NS_A, NS_B, NS_C)
            mix.ns_eps = NS_EPS
        if "deslice_topk" in spec:
            mix.deslice_topk = int(spec["deslice_topk"])
        if "deslice_threshold" in spec:
            mix.deslice_threshold = float(spec["deslice_threshold"])
        if spec.get("qwen_sdpa_gate"):
            mix.qwen_sdpa_gate = True
        if spec.get("res_gate"):
            b.use_res_gate = True
        if "recur_T" in spec:
            b.recur_T = int(spec["recur_T"])
    return model


def build(spec, dim, depth, slice_num, n_cls):
    if spec["kind"] == "patch":
        return PatchNet(dim, depth, spec["patch"], n_cls)
    m = SliceNet(dim, depth, slice_num, spec["norm"], n_cls,
                 spec.get("readout", "slices"), spec.get("local", False),
                 spec.get("n_freq", 0))
    apply_slice_flags(m, spec)
    return m

