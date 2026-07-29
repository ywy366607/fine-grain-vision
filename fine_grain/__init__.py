"""Fine-grain vision: content-adaptive slice pooling vs fixed patch grids.

Public API re-exports the pieces needed by training scripts and tests.
"""
from fine_grain.models import (
    ARMS,
    AdaTempSlice,
    AttnPool,
    Block,
    PatchNet,
    SelfAttn,
    SliceNet,
    apply_slice_flags,
    build,
    coords,
    deslice_support_size,
    newton_schulz,
    sparse_deslice_weights,
    NS_A,
    NS_B,
    NS_C,
    NS_EPS,
    NS_STEPS_DEFAULT,
)
from fine_grain.tasks import (
    BG,
    N_CLASSES,
    SIGNAL,
    TASKS,
    make_connect,
    make_glyph,
    make_kinks,
    make_lines,
    make_needle,
)
from fine_grain.train_utils import (
    _make_optimizer,
    collapse_stats,
    load_kinks_folder,
    pr_obj,
)

__version__ = "0.1.0"

__all__ = [
    "ARMS", "AdaTempSlice", "AttnPool", "Block", "PatchNet", "SelfAttn", "SliceNet",
    "TASKS", "apply_slice_flags", "build", "collapse_stats", "coords",
    "deslice_support_size", "make_connect", "make_glyph", "make_kinks",
    "make_lines", "make_needle", "newton_schulz", "pr_obj", "sparse_deslice_weights",
    "NS_A", "NS_B", "NS_C", "NS_EPS", "NS_STEPS_DEFAULT", "N_CLASSES",
]
