# Published metrics

JSON / text snapshots that back claims in the root README.

## Line-recon caveat (important)

`line_recon_64*.json` / RGB tables quoted in the showcase used a **legacy patch
decoder**: bilinear upsample of token features + 1×1 conv. That head **cannot**
emit independent per-pixel structure inside a 16×16 cell, so low patch Dice mixes

1. what the **encoder** fails to keep, and
2. what the **decoder** cannot unfold.

Current default in `scripts/line_recon.py` is the fair head:

```text
token → Linear(dim, p×p) → unpatchify
```

(`--patch-decoder bilinear` reproduces the old unfair setup.)

Re-run before citing new absolute patch Dice numbers:

```bash
python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600 --rgb \
  --patch-decoder unpatchify --out results/line_recon_64_fair.json
```

Spectrum figures in `present/figs/11_*.png` illustrate **patch mean**, not the
learned Conv2d stem used by `PatchNet`.

Regenerate other metrics with `scripts/train_benchmark.py` / `scripts/line_recon.py`.
Other raw logs under `results/` (if present locally) are gitignored.
