# Contributing

Issues and PRs welcome.

## Before a PR

1. Keep the public API in `fine_grain/` importable (`from fine_grain import build, ARMS`).
2. Run `python tests/test_deslice_scatter_and_gate.py`.
3. Prefer new experiments under `scripts/` with argparse + JSON under `results/` (gitignored except `published/`).
4. Do not commit `data/`, `checkpoints/`, or large raw logs.

## Scope

This repo is a **measurement package**, not a production VLM trainer. New tasks should state a falsifiable claim up front.
