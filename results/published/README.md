# Published metrics

JSON / text snapshots that back claims in the root README.

## Line recon — fair re-run (primary)

| File | Content |
|------|---------|
| [`line_recon_64_fair.json`](line_recon_64_fair.json) | Summary: slice vs patch16, **unpatchify** head, RGB |
| [`line_recon_64_fair_patch16.json`](line_recon_64_fair_patch16.json) | Full patch16 history (600 steps) |

| Arm | Head | Dice | IoU | Recall | notes |
|-----|------|------|-----|--------|-------|
| slice_loc_nogumbel | point | **1.000** | 1.000 | 1.000 | ~step 100 |
| patch16 | **unpatchify** | **0.215** | 0.121 | 0.880 | 600 steps |
| patch16 (legacy) | bilinear | ~0.153 | ~0.085 | ~0.685 | old published |

**Takeaway:** fixing the decoder lifts patch a little (0.15→0.22); slice remains near-perfect. Gap is **not only** an unfair head.

## Legacy files

`line_recon_64.json`, `line_recon_64_st.json`, `rgb_recon_fp.json` predate the fair head;
keep for history, prefer `line_recon_64_fair*.json` when citing patch Dice.

Spectrum figures in `present/figs/11_*.png` illustrate **patch mean**, not the
learned Conv2d stem used by `PatchNet`.

## Soft-deslice scatter (primary fix)

| File | Content |
|------|---------|
| [`scatter_ablation.json`](scatter_ablation.json) | RGB recon: soft vs top-k write vs gate (support size check) |
| [`scatter_ablation_gray.json`](scatter_ablation_gray.json) | Luma recon: harder FP regime |
| [`topk_deslice_ablation.json`](topk_deslice_ablation.json) | Earlier gray 400-step soft vs topk2 vs st+topk2 |

**Design (shipped in `fine_grain.models`):**

1. **Primary:** sparse deslice **write** (`deslice_topk` / threshold); soft mass **read** kept.
2. **Gate (correct placement):** Qiu et al. arXiv:2505.06708 — σ-gate **after** SDPA on slice tokens; optional residual-stream `x + σ(W x) ⊙ mix`.
3. **Fallback only:** `recur_T>1` shared-weight multi-pass (Universal-Transformer style), default off.

```bash
python scripts/scatter_ablation.py --rgb --steps 200 --amp
python tests/test_deslice_scatter_and_gate.py
```
