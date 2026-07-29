"""Synthetic fine-grained vision tasks (needle / glyph / lines / connect / kinks)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Signal colours (the needle) and distractor colours (the background blobs) are disjoint,
# so the label is recoverable only from the needle itself.
SIGNAL = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
BG = np.array([[.5, 0, .5], [0, .5, .5], [.45, .45, .45], [.35, .35, 0]], dtype=np.float32)
N_CLASSES = len(SIGNAL)


def _canvas(rng, n, R, n_blobs=3):
    """Low-frequency field + distractor blobs. Shared by every task so the haystack is
    identical and only the thing being looked for changes."""
    img = np.zeros((n, 3, R, R), np.float32)
    img += rng.normal(0, .05, (n, 3, 1, 1)).astype(np.float32)
    low = torch.from_numpy(rng.uniform(.1, .4, (n, 3, 4, 4)).astype(np.float32))
    img += F.interpolate(low, size=(R, R), mode="bilinear", align_corners=False).numpy()
    for _ in range(n_blobs):
        c = BG[rng.integers(0, len(BG), n)]
        sz = rng.integers(6, 17, n) * max(1, R // 32)
        yy, xx = rng.integers(0, R, n), rng.integers(0, R, n)
        for i in range(n):
            img[i, :, yy[i]:yy[i] + sz[i], xx[i]:xx[i] + sz[i]] = c[i][:, None, None]
    return img


def _done(img, lab, mask):
    """mask marks the label-bearing pixels; used only by the PR_obj diagnostic (P5')."""
    return (torch.from_numpy(np.clip(img, 0, 1)),
            torch.from_numpy(np.asarray(lab, np.int64)),
            torch.from_numpy(mask.reshape(len(mask), -1)))


def make_needle(rng, sizes, res=32, mult=1):
    """v1 task. Label = which of 4 SIGNAL colours the s x s square is. Colour is unique
    to the needle, so this rewards appearance-based aggregation -- possibly unfairly."""
    n, R = len(sizes), res * mult
    img, msk = _canvas(rng, n, R), np.zeros((n, R, R), bool)
    lab = rng.integers(0, N_CLASSES, n)
    for i in range(n):
        s = int(sizes[i]) * mult
        y, x = rng.integers(0, R - s), rng.integers(0, R - s)
        img[i, :, y:y + s, x:x + s] = SIGNAL[lab[i]][:, None, None]
        msk[i, y:y + s, x:x + s] = True
    return _done(img, lab, msk)


def _glyph_mask(kind, s):
    m = np.zeros((s, s), bool)
    c = s // 2
    if kind == 0:                                   # filled square
        m[:] = True
    elif kind == 1:                                 # cross
        m[c, :] = True; m[:, c] = True
    elif kind == 2:                                 # main diagonal
        np.fill_diagonal(m, True)
    else:                                           # L corner
        m[0, :] = True; m[:, 0] = True
    return m


def make_glyph(rng, sizes, res=32, mult=1):
    """Small-text proxy. Label = SHAPE.

    v1 OF THIS TASK WAS BROKEN (found 2026-07-29, first run): the glyph was drawn from the
    BG palette -- the same colours as the distractor blobs -- and glyph shape 0 is a filled
    square, which is exactly what a distractor blob is. So "which filled square is the
    glyph" had no answer: the target was not identifiable. Every arm collapsed to constant
    prediction (slice and slice_const returned bit-identical accuracy at all five sizes,
    the signature of a constant predictor), and 78/200 sampled images had the glyph colour
    also covering >5% of the background. That was a task bug, NOT a null result, and must
    not be cited as one.

    FIX: the colour comes from the SIGNAL palette, which is disjoint from the background,
    so the glyph is findable -- but it is drawn INDEPENDENTLY of the label, so colour says
    WHERE the object is and never WHAT it is. The colour shortcut is removed without making
    the object invisible."""
    n, R = len(sizes), res * mult
    img, msk = _canvas(rng, n, R), np.zeros((n, R, R), bool)
    lab = rng.integers(0, 4, n)
    col = SIGNAL[rng.integers(0, len(SIGNAL), n)]      # findable, label-independent
    for i in range(n):
        s = int(sizes[i]) * mult
        m = _glyph_mask(int(lab[i]), s)
        y, x = rng.integers(0, R - s), rng.integers(0, R - s)
        img[i, :, y:y + s, x:x + s][:, m] = col[i][:, None]
        msk[i, y:y + s, x:x + s] |= m
    return _done(img, lab, msk)


def make_lines(rng, sizes, res=32, mult=1):
    """Metro-map proxy at the DETECTION level. One straight line of width w spanning the
    whole image; label = which SIGNAL colour it is. Thin but EXTENDED: unlike a small
    blob, its area is spread over many tokens instead of being buried inside one."""
    n, R = len(sizes), res * mult
    img, msk = _canvas(rng, n, R), np.zeros((n, R, R), bool)
    lab = rng.integers(0, N_CLASSES, n)
    for i in range(n):
        w = int(sizes[i]) * mult
        p = int(rng.integers(0, R - w))
        if rng.random() < 0.5:
            img[i, :, p:p + w, :] = SIGNAL[lab[i]][:, None, None]; msk[i, p:p + w, :] = True
        else:
            img[i, :, :, p:p + w] = SIGNAL[lab[i]][:, None, None]; msk[i, :, p:p + w] = True
    return _done(img, lab, msk)


def make_connect(rng, sizes, res=32, mult=1):
    """Metro-map proxy at the TOPOLOGY level. Several same-coloured lines plus two marked
    dots; label = whether the dots sit on the SAME line (chance 0.5, binary). Colour is
    shared across lines so only geometry answers it. Exploratory -- see P8."""
    n, R = len(sizes), res * mult
    img, msk = _canvas(rng, n, R, n_blobs=1), np.zeros((n, R, R), bool)
    lab = np.zeros(n, np.int64)
    for i in range(n):
        w = int(sizes[i]) * mult
        rows = rng.choice(np.arange(1, R - w - 1), size=3, replace=False)
        for r in rows:
            img[i, :, r:r + w, :] = SIGNAL[0][:, None, None]; msk[i, r:r + w, :] = True
        same = rng.random() < 0.5
        lab[i] = int(same)
        r1 = rows[0]
        r2 = rows[0] if same else rows[1]
        x1, x2 = rng.integers(2, R - 2), rng.integers(2, R - 2)
        img[i, :, r1:r1 + w, x1] = 1.0                      # white dot on line r1
        img[i, :, r2:r2 + w, x2] = 0.0                      # black dot on line r2
    return _done(img, lab, msk)


def _clearance_ok(occ, ny, nx, tail, R):
    """New red (ny,nx) vs existing reds in its 3x3 (九宫格).

    A 90° corner needs 3 cells in one 九宫格: prev, tip, new. So the new cell may
    8-touch the last **two** path pixels (tail[-2], tail[-1]=tip). Any other red in
    the 九宫格 would mean two non-incident strokes merged (gap < 1 px).
    """
    if occ[ny, nx]:
        return False
    allowed = set(tail[-2:]) if len(tail) >= 2 else set(tail[-1:])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = ny + dy, nx + dx
            if not (0 <= yy < R and 0 <= xx < R) or not occ[yy, xx]:
                continue
            if (yy, xx) not in allowed:
                return False
    return True


def _run_len_clear(occ, tail, dy, dx, R, length):
    """Simulate `length` steps with 九宫格 isolation; tail is list of (y,x) path tips."""
    t = list(tail)
    for _ in range(length):
        y, x = t[-1]
        ny, nx = y + dy, x + dx
        if not (0 <= ny < R and 0 <= nx < R):
            return 0
        if not _clearance_ok(occ, ny, nx, t, R):
            return 0
        t.append((ny, nx))
    return length


def _draw_axis_seg_kinks(img, msk, occ, tail, dy, dx, length, color):
    """Extend path `tail` by `length` steps; returns updated tail or None."""
    R = img.shape[-1]
    if _run_len_clear(occ, tail, dy, dx, R, length) < length:
        return None
    t = list(tail)
    for _ in range(length):
        y, x = t[-1]
        ny, nx = y + dy, x + dx
        img[:, ny, nx] = color
        msk[ny, nx] = True
        occ[ny, nx] = True
        t.append((ny, nx))
    return t


def _sample_polyline_kinks(rng, R, n_kinks, color, total_len=None, max_tries=100):
    """Axis-aligned 1px polyline with exactly n_kinks intermediate 90° turns.

    Isolation (九宫格): a new red may only neighbour the path's last two pixels
    (前两点). That fits a 90° corner (3 cells in one 3x3) while keeping ≥1 empty
    pixel from any non-incident red.
    """
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n_seg = int(n_kinks) + 1
    if total_len is None:
        total_len = int(rng.integers(max(n_seg * 4, R // 2), max(n_seg * 5, int(R * 1.2)) + 1))
    base, rem = divmod(int(total_len), n_seg)
    seg_lens = [max(3, base + (1 if s < rem else 0)) for s in range(n_seg)]
    for s in range(n_seg):
        seg_lens[s] = max(3, int(round(seg_lens[s] * float(rng.uniform(0.85, 1.15)))))

    margin = max(4, R // 8)
    for _try in range(max_tries):
        img = np.zeros((3, R, R), np.float32)
        msk = np.zeros((R, R), bool)
        occ = np.zeros((R, R), bool)
        y = int(rng.integers(margin, R - margin))
        x = int(rng.integers(margin, R - margin))
        img[:, y, x] = color
        msk[y, x] = True
        occ[y, x] = True
        tail = [(y, x)]                 # path order; clearance uses last two
        prev_dir = None
        ok_all = True
        for s, length in enumerate(seg_lens):
            cands = []
            for d in dirs:
                if prev_dir is None:
                    cands.append(d)
                else:
                    if d == (-prev_dir[0], -prev_dir[1]) or d == prev_dir:
                        continue
                    cands.append(d)
            rng.shuffle(cands)
            placed = False
            for dy, dx in cands:
                L = length
                while L >= 3:
                    if _run_len_clear(occ, tail, dy, dx, R, L) == L:
                        break
                    L -= 1
                if L < 3:
                    continue
                new_tail = _draw_axis_seg_kinks(img, msk, occ, tail, dy, dx, L, color)
                if new_tail is None:
                    continue
                tail = new_tail
                prev_dir = (dy, dx)
                placed = True
                break
            if not placed:
                ok_all = False
                break
        if ok_all:
            return img, msk
    # short fallback
    img = np.zeros((3, R, R), np.float32)
    msk = np.zeros((R, R), bool)
    occ = np.zeros((R, R), bool)
    y = x = R // 2
    img[:, y, x] = color
    msk[y, x] = True
    occ[y, x] = True
    tail = [(y, x)]
    prev_dir = None
    for s in range(n_seg):
        for dy, dx in dirs:
            if prev_dir is not None and (
                    (dy, dx) == prev_dir or (dy, dx) == (-prev_dir[0], -prev_dir[1])):
                continue
            for L in range(min(seg_lens[s], 16), 2, -1):
                if _run_len_clear(occ, tail, dy, dx, R, L) == L:
                    new_tail = _draw_axis_seg_kinks(img, msk, occ, tail, dy, dx, L, color)
                    if new_tail is not None:
                        tail = new_tail
                        prev_dir = (dy, dx)
                        break
            else:
                continue
            break
    return img, msk


def _sample_polyline_kinks_in_tile(rng, R, n_kinks, color, tile=16, max_tries=120):
    """Hard mode: the entire polyline (all turns) lives inside one tile×tile window.

    With tile=16 this is one standard ViT patch cell: many 90° kinks + 1px clearance
    packed into a single patch footprint — patch16 has only one token for the whole
    structure. Returns (img3,HW, msk) in full-R canvas coordinates, or None.
    """
    tile = int(tile)
    if tile < 8 or tile > R:
        return None
    # place tile so full window is on-canvas
    y0 = int(rng.integers(0, R - tile + 1))
    x0 = int(rng.integers(0, R - tile + 1))
    # short ink: must fit in tile with 1px isolation; ~0.35*tile per segment budget
    n_seg = int(n_kinks) + 1
    total_len = int(rng.integers(n_seg * 3, max(n_seg * 3 + 1, int(0.9 * tile * tile / 8))))
    local, lmsk = _sample_polyline_kinks(
        rng, tile, n_kinks, color, total_len=total_len, max_tries=max_tries)
    if int(lmsk.sum()) < n_seg * 2 + 1:
        return None
    img = np.zeros((3, R, R), np.float32)
    msk = np.zeros((R, R), bool)
    img[:, y0:y0 + tile, x0:x0 + tile] = local
    msk[y0:y0 + tile, x0:x0 + tile] = lmsk
    return img, msk, (y0, x0, tile)


def make_kinks(rng, sizes, res=32, mult=1, hard_tile=0, hard_frac=0.0):
    """Count 90° kinks on a 1px red polyline in a noisy (non-red) background.

    Label = kink count mapped to class (default k=5..10 → lab = k-5, 6 classes).

    hard_tile>0 and hard_frac in (0,1]: with that probability pack the whole polyline
    into one hard_tile×hard_tile window (e.g. 16 = one ViT patch). Killer samples for
    patch16: many turns in one token footprint.
    """
    n, R = len(sizes), res * mult
    n_blobs = 5 if R <= 64 else 8
    img = _canvas(rng, n, R, n_blobs=n_blobs)
    msk = np.zeros((n, R, R), bool)
    lab = np.zeros(n, np.int64)
    red = SIGNAL[0]
    k_min_global = 5
    for i in range(n):
        k = int(sizes[i])
        k = max(1, min(32, k))
        use_hard = hard_tile > 0 and float(rng.random()) < float(hard_frac)
        pm = None
        if use_hard:
            for _ in range(24):
                got = _sample_polyline_kinks_in_tile(rng, R, k, red, tile=hard_tile)
                if got is not None:
                    _poly, pm, _box = got
                    break
        if pm is None:
            lo = max(k * 6, int(0.6 * R))
            hi = max(lo + 1, int(1.4 * R))
            total_len = int(rng.integers(lo, hi + 1))
            _poly, pm = _sample_polyline_kinks(rng, R, k, red, total_len=total_len)
        img[i] = np.clip(img[i], 0, 1)
        for c in range(3):
            img[i, c][pm] = red[c]
        msk[i] = pm
        lab[i] = max(0, k - k_min_global)
    return _done(img, lab, msk)


TASKS = {
    "needle":  dict(fn=make_needle,  sizes=[1, 2, 3, 4, 6, 8], n_cls=4),
    "glyph":   dict(fn=make_glyph,   sizes=[3, 4, 6, 8, 12],   n_cls=4),
    "lines":   dict(fn=make_lines,   sizes=[1, 2, 3, 4],       n_cls=4),
    "connect": dict(fn=make_connect, sizes=[1, 2, 3],          n_cls=2),
    # 1px red polyline kink count. k=5..10 → 6 classes (lab = k-5). Standard ViT
    # scale experiments use --res 256 --arms patch16 + ours.
    "kinks":   dict(fn=make_kinks,   sizes=[5, 6, 7, 8, 9, 10], n_cls=6),
}

