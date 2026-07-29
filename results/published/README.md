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
