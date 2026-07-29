# Fine-Grain Vision

**Content-adaptive slice pooling vs fixed patch grids** for fine-grained visual structure.

Self-contained PyTorch experiments. No external datasets required — all tasks are synthetic and reproducible.

> Architectural comparison at tiny scale (depth 3, dim 64). Numbers are **not** comparable to production VLMs (MoonViT, Qwen-VL, …).

## Question

Modern VLM vision stacks remove *resize* but keep a **fixed patch grid**. Is that grid what buries small structure, or is the real bottleneck effective resolution / re-acquisition?

This repo measures both sides under a matched token budget:

| Arm | Mechanism |
|-----|-----------|
| `patch4` / `patch8` / `patch16` | ViT-style patchify + MHSA (baseline; deliberately **not** weakened) |
| `slice` / `slice_loc_nogumbel` | Transolver++-style soft assignment + mass-normalized pooling |
| `slice_*_st` | + Stiefel directions via Newton–Schulz (Muon coefficients) |
| controls | `slice_const`, `slice_sum`, Gumbel on/off, sparse deslice write |

## Key findings (pre-registered where noted)

| ID | Result |
|----|--------|
| **P1/P4** | Patch accuracy tracks **s/p** (object size / patch size), not absolute s. Raising resolution buys fine grain at 4× tokens. |
| **P6** | Thin **lines** survive patch grids; **point-like** needles do not — fine ≠ small. |
| **P7** | **Mass normalization** is the size-invariance mechanism (not per-pixel tokens alone). |
| **Oracle** | Glyph memorization failure is largely **optimization** (Gumbel noise), not expressivity. |
| **Line recon (fair)** | RGB pure-red 1px polylines, res 64, 600 steps. **slice** Dice **1.00** by ~step 100. **patch16** with fair head `Linear→unpatchify`: Dice **0.215** (IoU 0.12, recall 0.88). Legacy bilinear head was **~0.15** — fair head helps a little, **does not close the gap**. Stiefel is mainly **anti-collapse**, not recon magic. See [`results/published/line_recon_64_fair.json`](results/published/line_recon_64_fair.json). |
| **Caveats** | (1) Prefer **RGB** over luma (luma≈0.299 shortcut on red lines). (2) Default patch recon head is unpatchify; `--patch-decoder bilinear` is legacy only. (3) Spectrum wipe figure uses **patch mean** as illustration — real stem is learned `Conv2d(k=p,s=p)`; claim is **capacity/SNR falls with s/p**, not “inevitably wipes 1px lines.” |

Published JSON snapshots live in [`results/published/`](results/published/).

## Layout

```text
fine_grain/           # installable package
  models.py           # AdaTempSlice, SliceNet, PatchNet, ARMS registry
  tasks.py            # needle / glyph / lines / connect / kinks generators
  train_utils.py      # optim, pool, collapse probes
scripts/
  train_benchmark.py  # classification sweeps (main benchmark)
  line_recon.py       # dense polyline mask recon
  build_kinks_dataset.py
tests/
present/              # HTML showcase + figures
reference/            # notes on Transolver++ / MoonViT
results/published/    # key metrics for the README claims
```

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
# optional editable install
pip install -e .
```

GPU (CUDA) is optional but recommended for full sweeps.

## Quickstart

```bash
# unit tests (CPU, seconds)
python tests/test_deslice_scatter_and_gate.py
# or: pytest -q

# needle size sweep (tiny default steps for smoke)
python scripts/train_benchmark.py --task needle --arms patch4,slice --steps 50 --seeds 1

# glyph shape task
python scripts/train_benchmark.py --task glyph --arms patch4,slice_loc_nogumbel --steps 1500

# line reconstruction @ 64^2 (fair patch head = unpatchify by default)
python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600 --rgb
# legacy unfair patch head (bilinear + 1x1), for comparing old numbers only
python scripts/line_recon.py --arms patch16 --patch-decoder bilinear --rgb

# optional: materialize kinks dataset to disk
python scripts/build_kinks_dataset.py --out data/kinks256 --n_train 6000 --n_val 600
```

Speed micro-bench (CUDA):

```bash
python scripts/train_benchmark.py --bench --device cuda
```

## Tasks

| Task | Label | What it stresses |
|------|-------|------------------|
| `needle` | 4 signal colors | Point-like objects vs patch size |
| `glyph` | 4 shapes (color-matched to BG) | Local shape without color shortcut |
| `lines` | line color | Extended 1–4 px structures |
| `connect` | same-polyline? | Topology (often near chance at this scale) |
| `kinks` | kink count 5–10 | Thin red polyline geometry |
| line recon | dense mask | Pixel-level recovery of the polyline |

## Showcase

Open [`present/showcase.html`](present/showcase.html) in a browser for figures (needle curves, spectrum wipe, line-recon gallery).

## Design notes

- **Baseline fidelity (classification):** patch arms get no extras (no RoPE freebies). `patch4` is *stronger* than real MoonViT (which pixel-shuffles 2×2 before projection) — intentional for classification sweeps.
- **Line-recon head (dense):** slice uses per-point logits (full HxW). Patch default is **unpatchify** so within-patch structure is expressible; do not cite bilinear-head Dice as pure encoder failure.
- **Patchify ≠ mean pool:** stem is `Conv2d(3 → dim-2, kernel=p, stride=p)` — a learned linear map over the p²×3 window. Mean pooling is only a special case; high-frequency directions *can* survive in the subspace of the kernel. The hard claim is **s/p-limited capacity/SNR and addressing**, not spectral annihilation.
- **Slice stack:** mass-norm soft pool (read) + optional sparse deslice write; optional stride-1 DW 3×3 for locality without stride-downsampling.
- **Not in scope:** real OCR/characters (glyph = 4 geometric shapes only), saccadic re-acquisition, multi-hop map reasoning, production VLM training.

## Citation / lineage

- Transolver++ physics attention: [thuml/Transolver_plus](https://github.com/thuml/Transolver_plus) (ICML 2025)
- Newton–Schulz Stiefel / Muon coefficients: PyTorch `torch.optim._muon`
- Qwen gated attention (optional residual / SDPA gate): Qiu et al., arXiv:2505.06708

## License

MIT — see [LICENSE](LICENSE).
