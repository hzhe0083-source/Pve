#!/usr/bin/env python3
"""Figure 1 (final): concept A plus both judges' fixes and the grafts that fit. Double column, 7.16 x 2.70 in.

(a) schematic of the null-space adapter, with the read-space note and the prefix-sum ablation (P = I) defined at P;
(b) PlaceCube learning curves (median and IQR over 8 seeds, each seed's last value carried forward) with the
    steps-to-90% median and min-max range per arm;
(c) Pick success after learning Place, one labelled row per arm, with the Pick head reading the shared h'.

Reads ../data (read-only) and action_change_exact_check.csv next to this script. Writes fig1.pdf, fig1.svg and fig1.png
(300 dpi) next to this script, prints every number shown with its source, and runs a layout check: no text/text or
text/marker overlap, every marker inside its axes, everything on the page, no text below 7 pt.
"""
import csv
import itertools
import json
import logging
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG1 = os.path.dirname(HERE)
DATA = os.path.join(FIG1, "data")
RUNS = os.path.join(DATA, "a", "protect_np", "runs")
RET_A = os.path.join(DATA, "a", "protect_np", "ret")
RET_C = os.path.join(DATA, "c", "protect_c", "ret")
EXACT = os.path.join(HERE, "action_change_exact_check.csv")
SESSION_LOG = "~/.claude/projects/-home-ryan-Documents-pve/2a31a520-ee40-44e4-b92d-86ed12830b96.jsonl:15322"
EVAL_EVERY = 102400
SEEDS = range(1, 9)

# ---------------------------------------------------------------- style tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # ours, prefix-sum zero-init, prefix-sum matched init
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"          # text primary / secondary, muted gray
AXIS = "#c3c2b7"                                            # axis / baseline hairline
REF = "#8a8882"                                             # reference series (full fine-tuning, earlier code)
FROZEN_FC, FROZEN_EC = "#e6e5e0", "#9b9991"                 # frozen parts (neutral gray)
TRAIN_FC = "#cde2fb"                                        # trainable parts: blue ramp step 100 + blue edge
SURFACE = "#ffffff"
GUIDE = "#e7e6e0"                                           # hairline row guides in (c)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.labelsize": 7,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

NUMBERS = []    # (what, value, source) for every number drawn


def note(what, value, source):
    NUMBERS.append((what, value, source))


def rel(p):
    return os.path.relpath(p, FIG1) if p.startswith(FIG1) else p


# ---------------------------------------------------------------- data: learning curves and steps to 90%
def load_eval(task, arm, seed):
    path = os.path.join(RUNS, f"{task}_{arm}_s{seed}", "eval.csv")
    steps, succ = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            succ.append(float(row["success_once"]))
    steps = np.array(steps)
    assert (steps == np.arange(len(steps)) * EVAL_EVERY).all(), path      # evals every 102,400 steps from 0
    return steps, np.array(succ), path


def t90(steps, succ, thr=0.9, k=3):
    """First step of k consecutive evals >= thr (the stopping rule and the pre-registered metric)."""
    for i in range(len(succ) - k + 1):
        if (succ[i:i + k] >= thr).all():
            return int(steps[i])
    return None


def carry_forward(curves):
    """Seeds x evals matrix; a run that stopped early keeps its last value."""
    n = max(len(y) for _, y in curves)
    m = np.empty((len(curves), n))
    for i, (_, y) in enumerate(curves):
        m[i, :len(y)] = y
        m[i, len(y):] = y[-1]
    return np.arange(n) * EVAL_EVERY, m


def signed_rank_p_one_sided(d):
    """Exact one-sided Wilcoxon signed-rank p (H1: d > 0) by enumerating all 2^n sign flips; mid-ranks for ties."""
    d = np.asarray(d, float)
    d = d[d != 0]
    a = np.abs(d)
    order = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):                       # mid-ranks
        ranks[a == v] = ranks[a == v].mean()
    w_obs = ranks[d > 0].sum()
    ws = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=len(a))]
    return np.mean(np.array(ws) >= w_obs - 1e-12)


ARMS = [("nsa64", "ours", BLUE), ("nsa64np", "zero-init", ORANGE), ("nsa64npr", "matched init", AQUA)]
MARK = {"nsa64": "o", "nsa64np": "s", "nsa64npr": "D", "monitor": "o"}   # shape = secondary encoding
MSIZE = {"o": 4.8, "s": 4.3, "D": 4.0}
EXPECTED_T90 = [0.61, 0.72, 0.61, 0.61, 0.61, 0.61, 0.72, 0.61]          # ours, from the brief, M steps

curves, t90s = {}, {}
for arm, _, _ in ARMS:
    cs, ts = [], []
    for s in SEEDS:
        st, y, path = load_eval("place", arm, s)
        cs.append((st, y))
        ts.append(t90(st, y))
        assert ts[-1] is not None, path
    curves[arm] = carry_forward(cs)
    t90s[arm] = np.array(ts)
assert [round(t / 1e6, 2) for t in t90s["nsa64"]] == EXPECTED_T90
assert np.median(t90s["nsa64"]) == 6 * EVAL_EVERY
assert np.median(t90s["nsa64np"]) == 9 * EVAL_EVERY and np.median(t90s["nsa64npr"]) == 9 * EVAL_EVERY
PVAL, GAPS = {}, {}
for arm in ("nsa64np", "nsa64npr"):       # every paired gap >= 2 eval intervals; pre-registered exact test
    gaps = (t90s[arm] - t90s["nsa64"]) // EVAL_EVERY
    assert (gaps >= 2).all(), (arm, gaps)
    GAPS[arm] = gaps
    PVAL[arm] = signed_rank_p_one_sided(gaps)
    assert abs(PVAL[arm] - 1 / 256) < 1e-12, PVAL[arm]
N_PAIRS = sum(len(g) for g in GAPS.values())
N_FASTER = sum(int((g > 0).sum()) for g in GAPS.values())
assert N_PAIRS == N_FASTER == 16

STACK_MED = {}
for arm, _, _ in ARMS:                    # caption only
    STACK_MED[arm] = np.median([t90(*load_eval("stack", arm, s)[:2]) for s in (1, 2, 3)]) / 1e6
assert [round(STACK_MED[a], 2) for a in ("nsa64", "nsa64np", "nsa64npr")] == [1.23, 2.46, 2.87], STACK_MED

# ---------------------------------------------------------------- data: exactness (Pick action change)
with open(EXACT) as f:
    rows_x = list(csv.DictReader(line for line in f if not line.startswith("#")))
ours_x = [r for r in rows_x if r["arm"] == "nsa64"]
ps_x = [r for r in rows_x if r["arm"] in ("nsa64np", "nsa64npr")]
assert len(ours_x) == 8 and len(ps_x) == 16
X_STORED_MAX = max(float(r["stored"]) for r in ours_x)
X_STORED_MIN = min(float(r["stored"]) for r in ours_x)
X_RANDOM_MAX = max(float(r["random"]) for r in ours_x)
X_NOPROJ = (min(float(r["raw_no_projection"]) for r in ours_x), max(float(r["raw_no_projection"]) for r in ours_x))
X_PS = (min(float(r["stored"]) for r in ps_x), max(float(r["stored"]) for r in ps_x))
assert X_STORED_MAX == 1.5e-6 and X_STORED_MIN == 1.2e-6 and X_RANDOM_MAX == 2.1e-6
assert X_NOPROJ == (0.52, 0.83) and X_PS == (0.70, 0.92)
assert all(float(r["own"]) == 0.0 for r in rows_x)


# ---------------------------------------------------------------- data: retention (Pick, 256 deterministic episodes)
def final_ckpt(path, key):
    with open(path) as f:
        d = json.load(f)
    finals = [v for k, v in d["runs"].items() if k.endswith("final_ckpt.pt")]
    assert len(finals) == 1, path
    v = finals[0]
    if key == "old_head_last":
        assert v.get("read") == "own", path        # prefix-sum: old_head = own level, old_head_last = shared vector
        assert v["old_head"] == d["old_expert"], path     # through its own level, prefix-sum keeps Pick exact
    return v[key], d["old_expert"]


ret = {}
expert = None
for arm, key in (("nsa64", "old_head"), ("nsa64np", "old_head_last"), ("nsa64npr", "old_head_last")):
    vals, srcs = [], []
    for s in SEEDS:
        p = os.path.join(RET_A, f"place_{arm}_s{s}.json")
        if os.path.exists(p):
            v, e = final_ckpt(p, key)
            vals.append(v)
            srcs.append(p)
            expert = e if expert is None else expert
            assert e == expert
    ret[arm] = (np.array(vals), srcs, key)
ft_vals, ft_srcs = [], []
for s in (1, 2, 3):
    p = os.path.join(RET_C, f"place_monitor_s{s}.json")
    v, e = final_ckpt(p, "old_head")
    assert e == expert
    ft_vals.append(v)
    ft_srcs.append(p)
ret["monitor"] = (np.array(ft_vals), ft_srcs, "old_head")

# ---------------------------------------------------------------- figure geometry (inches)
FW, FH = 7.16, 2.70
fig = plt.figure(figsize=(FW, FH))
RICH_GROUP = {}          # text artist -> id of the rich() call it belongs to (runs of one call may touch)


def axes_in(l, b, w, h, **kw):
    return fig.add_axes([l / FW, b / FH, w / FW, h / FH], **kw)


def ftext(x, y, s, **kw):
    """Text at a figure position given in inches."""
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", 7)
    return fig.text(x / FW, y / FH, s, **kw)


def run_width(s, size, style="normal", weight="normal"):
    tt = fig.text(0, 0, s, fontsize=size, fontstyle=style, fontweight=weight)
    w = tt.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    tt.remove()
    return w


def rich(x, y, parts, ha="left", color=INK, z=5):
    """Runs on one baseline at figure position (x, y) in inches, DejaVu Sans only (no mathtext, so no Computer
    Modern fallback glyphs and no sub/superscripts below 7 pt). parts: numbers are gaps in points; tuples are
    (text, size, style, baseline shift in points) with an optional 5th item, the colour. Returns the width (in)."""
    items = []
    for p in parts:
        if isinstance(p, (int, float)):
            items.append((None, p / 72))
        else:
            items.append((p, run_width(p[0], p[1], p[2])))
    total = sum(w for _, w in items)
    x0 = x - (total / 2 if ha == "center" else total if ha == "right" else 0)
    gid = len(RICH_GROUP) + 1
    for it, w in items:
        if it is not None:
            txt, size, style, dy = it[:4]
            col = it[4] if len(it) > 4 else color
            # SVG viewers drop leading/trailing spaces of a <text> (svg.fonttype='none'), so a run's leading
            # spaces become an x offset and its trailing spaces are not drawn (the next run is placed by width)
            core = txt.strip(" ")
            lead = run_width(txt, size, style) - run_width(txt.lstrip(" "), size, style) if txt[:1] == " " else 0.0
            tt = fig.text((x0 + lead) / FW, (y + dy / 72) / FH, core, fontsize=size, fontstyle=style, color=col,
                          ha="left", va="baseline", zorder=z)
            RICH_GROUP[tt] = gid
        x0 += w
    return total


def sym(base, sub=None, size=10, sub_size=7, drop=2.3):
    """An italic symbol with an upright subscript, as rich() parts."""
    out = [(base, size, "italic", 0)]
    if sub is not None:
        out += [0.4, (sub, sub_size, "normal", -drop)]
    return out


TITLE_Y = 2.575
XLAB_Y = 0.13                           # shared baseline of the (b) and (c) x labels
STRIP_B = 0.42                          # bottom of the (b) strip and of (c): their tick labels share a row
# ================================================================ (a) schematic
AW = 3.30
ax = axes_in(0.0, 0.0, AW, FH)
ax.set_xlim(0, AW)
ax.set_ylim(0, FH)
ax.set_aspect("equal")
ax.axis("off")

LW = 0.9           # connector width (pt)
HEAD = 5.5         # arrow head scale


def box(cx, cy, w, h, fc, ec, lw=0.8, r=0.035, z=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def path(pts, color=INK2, lw=LW, arrow=True, z=2):
    xs_, ys_ = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs_[:-1], ys_[:-1], color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=z)
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=HEAD), zorder=z)
    else:
        ax.plot(xs_[-2:], ys_[-2:], color=color, lw=lw, solid_capstyle="butt", zorder=z)


def junction(x, y, color):
    ax.add_patch(Circle((x, y), 0.021, fc=color, ec="none", zorder=z_j))


z_j = 4


def t(x, y, s, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", 7)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 5)
    return ax.text(x, y, s, **kw)


yT, yA = 1.80, 1.02                     # trunk row, adapter row
yM = (yT + yA) / 2
# input (light edge: the only white box with a dark border is the fixed operator P)
xin = 0.20
box(xin, yM, 0.28, 0.26, "white", FROZEN_EC, lw=0.6)
rich(xin, yM - 0.036, sym("x", size=10), ha="center")
t(xin, yM - 0.20, "state", color=INK2, va="top")
t(xin, yM - 0.31, "(42-d)", color=INK2, va="top")
xf = 0.46                                # fork of x into trunk and adapter
path([(xin + 0.14, yM), (xf, yM)], arrow=False)
path([(xf, yM), (xf, yT), (0.58, yT)])
junction(xf, yM, INK2)
# trunk: three frozen tanh layers of 256
tx = [0.71, 1.07, 1.43]
tw, th = 0.26, 0.28
for cx, sub in zip(tx, ["1", "2", None]):
    box(cx, yT, tw, th, FROZEN_FC, FROZEN_EC)
    rich(cx, yT - 0.045, sym("h", sub), ha="center")
for a, b in zip(tx[:-1], tx[1:]):
    path([(a + tw / 2, yT), (b - tw / 2, yT)])
t(1.07, yT + 0.355, "frozen Pick trunk", color=INK2)
t(1.07, yT + 0.235, "3 tanh layers, 256 units", color=INK2)
# sum node
xs = 1.78
ax.add_patch(Circle((xs, yT), 0.07, fc="white", ec=INK, lw=0.9, zorder=4))
t(xs, yT - 0.004, "+", fontsize=8.5)
path([(tx[-1] + tw / 2, yT), (xs - 0.07, yT)])
# h' and the two heads
xj = 2.00
path([(xs + 0.07, yT), (xj, yT)], color=INK, arrow=False)
rich((xs + 0.07 + xj) / 2, yT + 0.045, [("h", 9, "italic", 0), ("′", 9, "normal", 0)], ha="center")
hx, hw, hh = 2.35, 0.46, 0.24
for dy, fc, ec, sub in ((0.30, FROZEN_FC, FROZEN_EC, "old"), (-0.30, TRAIN_FC, BLUE, "new")):
    path([(xj, yT), (xj, yT + dy), (hx - hw / 2, yT + dy)], color=INK)
    box(hx, yT + dy, hw, hh, fc, ec, lw=0.8 if ec != BLUE else 1.0)
    rich(hx, yT + dy - 0.045, sym("W", sub), ha="center")
    path([(hx + hw / 2, yT + dy), (hx + hw / 2 + 0.13, yT + dy)], color=INK)
junction(xj, yT, INK)
ox = hx + hw / 2 + 0.17
t(ox, yT + 0.30 + 0.06, "Pick", ha="left")
t(ox, yT + 0.30 - 0.07, "8 actions", ha="left", color=INK2)
t(ox, yT - 0.30 + 0.06, "Place", ha="left")
t(ox, yT - 0.30 - 0.07, "8 actions", ha="left", color=INK2)
t(xj + 0.03, yT - 0.30 - 0.20, "init: copy of Pick head", color=INK2, ha="left")
# adapter g(x, h2): Linear 298->64, tanh, Linear 64->256 (bottleneck shape)
ax0, ax1, half, waist = 0.92, 1.52, 0.15, 0.05
axm = (ax0 + ax1) / 2
ax.add_patch(Polygon([(ax0, yA - half), (axm, yA - waist), (ax1, yA - half), (ax1, yA + half), (axm, yA + waist),
                      (ax0, yA + half)], closed=True, fc=TRAIN_FC, ec=BLUE, lw=1.0, zorder=3))
path([(xf, yM), (xf, yA), (ax0, yA)])
top_at = yA + half - (tx[1] - ax0) / (axm - ax0) * (half - waist)
path([(tx[1], yT - th / 2), (tx[1], top_at + 0.005)])
G = 8
rich(axm, yA - half - 0.135, [("adapter", 7, "normal", 0), 2.2, ("g", G, "italic", 0), ("(", G, "normal", 0),
                               ("x", G, "italic", 0), (", ", G, "normal", 0), ("h", G, "italic", 0), 0.3,
                               ("2", 7, "normal", -1.9), (")", G, "normal", 0)], ha="center")
t(axm, yA - half - 0.245, "298 → 64 → 256", color=INK2)
# projector P (fixed operator)
px, ps = xs, 0.24
path([(ax1, yA), (px - ps / 2, yA)], color=BLUE, lw=1.1)
ax.add_patch(Rectangle((px - ps / 2, yA - ps / 2), ps, ps, fc="white", ec=INK, lw=1.5, zorder=4))
t(px, yA, "P", fontsize=9.5, fontweight="bold")
path([(px, yA + ps / 2), (px, yT - 0.07)], color=BLUE, lw=1.1)
rich(px - 0.05, yM - 0.01, [("Pg", 8.5, "italic", 0)], ha="right")
lx = px + ps / 2 + 0.08
F = 8.5
rich(lx, yA + 0.04, [("P", F, "italic", 0), 2.5, ("=", F, "normal", 0), 2.5, ("I", F, "italic", 0), 2.5,
                     ("−", F, "normal", 0), 2.5, ("B", F, "italic", 0), 0.6, ("T", 7, "normal", 3.4), 0.6,
                     ("B", F, "italic", 0), 5, ("(fixed)", 7, "normal", 0)])
# notes under the equation (baselines; the W_old line needs room for its subscript)
rich(lx, yA - 0.105, [("B", 7.5, "italic", 0), (": the 8 of 256 directions", 7, "normal", 0)], color=INK2)
rich(lx, yA - 0.235, sym("W", "old", 7.5, 7, 1.5) + [(" reads (one SVD)", 7, "normal", 0)], color=INK2)
rich(lx, yA - 0.368, [("prefix-sum ablation: ", 7, "normal", 0), ("P", 7.5, "italic", 0),
                      (" = ", 7, "normal", 0), ("I", 7.5, "italic", 0)], color=INK2)
# the guarantee
gy = 0.43
S = 9
rich(0.05, gy, sym("W", "old", S, 7, 2.0) + [2.5, ("P", S, "italic", 0), 3, ("=", S, "normal", 0), 3,
                                               ("0", S, "normal", 0), 7, ("⇒", S, "normal", 0), 7]
     + sym("W", "old", S, 7, 2.0) + [2.5, ("h", S, "italic", 0), ("′", S, "normal", 0), 3,
                                       ("=", S, "normal", 0), 3] + sym("W", "old", S, 7, 2.0)
     + [2.5, ("h", S, "italic", 0)])
t(0.05, gy - 0.155, "Pick actions unchanged at every state; no old data or replay", ha="left", va="baseline")
mant = f"{X_STORED_MAX / 1e-6:.1f}"
rich(0.05, gy - 0.31, [(f"measured max |Δa| ≤ {mant} × 10", 7, "normal", 0), ("−6", 7, "normal", 2.8),
                       (" over 102,400 stored Pick states", 7, "normal", 0)], color=INK2)
note("(a) max |Δa| of Pick actions, Pick head reads h' (max over the 8 final nsa64 ckpts of the per-run max over "
     "102,400 stored Pick states; per-run 1.2-1.5e-6)", f"{X_STORED_MAX:.1e} (printed '{mant} × 10^-6')",
     f"{rel(EXACT)} column 'stored', rows nsa64 (= exact_check.py printout, {SESSION_LOG})")
note("(a) stored Pick states in that check", "102,400", f"{rel(EXACT)} header (= printout: 'stored Pick states: "
     "(102400, 42)')")
note("(a) state size / layer widths / adapter shape / action count / directions W_old reads",
     "42; 3 x 256; 298->64->256; 8; 8 of 256",
     "architecture in the brief; matches /home/ryan/Documents/pve/wrwl/nsa.py (build_actor, NullSpaceTrunk)")
# title and key
ftext(0.02, TITLE_Y, "(a) Null-space adapter", fontsize=8, fontweight="bold", va="baseline")
kx, ky = 0.05, 2.40
for fc, ec, lw, lab in ((FROZEN_FC, FROZEN_EC, 0.8, "frozen"), (TRAIN_FC, BLUE, 1.0, "trained (PPO)"),
                        ("white", INK, 1.5, "fixed operator")):
    ax.add_patch(Rectangle((kx, ky - 0.05), 0.15, 0.10, fc=fc, ec=ec, lw=lw, zorder=4))
    tt = t(kx + 0.21, ky, lab, ha="left", color=INK2)
    kx += 0.21 + tt.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.17   # 1 data unit = 1 in

# ================================================================ (b) learning curves + steps-to-90% strip
BL, BW = 3.86, 1.40
B_BOT, B_TOP = 1.07, 2.03
axb = axes_in(BL, B_BOT, BW, B_TOP - B_BOT)
axs = axes_in(BL, STRIP_B, BW, 0.36, sharex=axb)
XMAX = 1.47
for arm, lab, col in ARMS[::-1]:
    grid, m = curves[arm]
    x = grid / 1e6
    med = np.median(m, axis=0)
    q1, q3 = np.percentile(m, [25, 75], axis=0)
    axb.fill_between(x, q1, q3, color=col, alpha=0.14, lw=0, zorder=1)
    axb.plot(x, med, color=col, lw=1.5, solid_capstyle="round", solid_joinstyle="round",
             zorder=3 if arm == "nsa64" else 2)
axb.axhline(0.9, color=MUTED, lw=0.6, zorder=0)
axb.text(0.02, 0.915, "90%", color=INK2, ha="left", va="bottom")
axb.set_xlim(0, XMAX)
axb.set_ylim(-0.02, 1.04)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(["0", "0.5", "1"])
axb.set_ylabel("Place success rate", labelpad=3)
axb.tick_params(axis="x", labelbottom=False, length=0)
for sp in ("top", "right", "bottom"):
    axb.spines[sp].set_visible(False)
axb.text(0.37, 0.62, "ours", ha="right", va="center")
axb.text(0.84, 0.40, "prefix-sum", ha="left", va="center")
note("(b) 90% threshold line", "0.9", "success benchmark (learned = 90%); stop_success 0.9 in args.json")
for arm, lab, col in ARMS:
    src = rel(os.path.join(RUNS, f"place_{arm}_s{{1..8}}", "eval.csv"))
    grid, m = curves[arm]
    note(f"(b) {lab}: median curve + IQR band over 8 seeds (carry-forward; curve ends at {grid[-1] / 1e6:.2f}M)",
         "per-eval success_once", src)

rows = {"nsa64": 2, "nsa64np": 1, "nsa64npr": 0}
for arm, lab, col in ARMS:
    v = t90s[arm] / 1e6
    y = rows[arm]
    mk = MARK[arm]
    axs.plot([v.min(), v.max()], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    axs.plot([np.median(v)], [y], mk, ms=MSIZE[mk] + 0.4, mfc=col, mec="white", mew=0.9, zorder=3)
    axs.text(v.min() - 0.07, y, f"{np.median(v):.2f}M", ha="right", va="center")
    src = rel(os.path.join(RUNS, f"place_{arm}_s{{1..8}}", "eval.csv"))
    note(f"(b) {lab}: median t90", f"{np.median(t90s[arm]) / 1e6:.4f}M, printed {np.median(v):.2f}M", src)
    note(f"(b) {lab}: t90 range bar (min-max over 8 seeds)", f"{v.min():.4f}-{v.max():.4f}M", src)
axs.set_ylim(-0.6, 2.6)
axs.set_yticks([2, 1, 0])
axs.set_yticklabels(["ours", "zero-init", "matched init"])
axs.tick_params(axis="y", length=0, labelcolor=INK)
axs.set_xticks([0, 0.5, 1.0])
axs.set_xticklabels(["0", "0.5", "1.0"])
ftext(BL + BW / 2, XLAB_Y, "env steps (M)", ha="center", va="baseline")
for sp in ("top", "right", "left"):
    axs.spines[sp].set_visible(False)
ftext(BL, 0.955, "steps to 90%, 8 seeds", color=INK2, va="center")
ftext(BL, 0.845, f"ours faster in {N_FASTER}/{N_PAIRS} seed pairs", color=INK2, va="center")
note("(b) seeds per arm (strip header)", "8", "place_<arm>_s1..8 eval.csv")
note("(b) seed pairs in which ours reaches 90% first (8 seeds x 2 prefix-sum variants; every gap >= 2 evals)",
     f"{N_FASTER}/{N_PAIRS}", "t90 per seed from the place eval.csv files above")
ftext(3.42, TITLE_Y, "(b) Learning Place", fontsize=8, fontweight="bold", va="baseline")

# legend for (b) (and the colours of (c), whose rows carry their own labels)
LX = 3.42
r1, r2, r3 = 2.44, 2.30, 2.16


def key_mark(x, y, col, mk, line=True):
    if line:
        fig.add_artist(Line2D([x / FW, (x + 0.24) / FW], [y / FH, y / FH], color=col, lw=1.5,
                              solid_capstyle="round"))
    fig.add_artist(Line2D([(x + 0.12) / FW], [y / FH], marker=mk, ms=MSIZE[mk], mfc=col, mec="white", mew=0.8))


key_mark(LX, r1, BLUE, "o")
ftext(LX + 0.30, r1, "Null-space adapter (ours)", va="center")
ftext(LX, r2, "Prefix-sum adapter (no projection):", va="center")
key_mark(LX, r3, ORANGE, "s")
ftext(LX + 0.30, r3, "zero-init", va="center")
x3 = LX + 0.30 + run_width("zero-init", 7) + 0.16
key_mark(x3, r3, AQUA, "D")
ftext(x3 + 0.30, r3, "matched init", va="center")

# ================================================================ (c) retention: one row per arm
CT = 5.42
C_ROWS = [("nsa64", 3, BLUE, "ours"), ("nsa64np", 2, ORANGE, "zero-init"), ("nsa64npr", 1, AQUA, "matched init"),
          ("monitor", 0, REF, "full FT†")]
LABW = max(run_width(lab, 7) for *_, lab in C_ROWS)
CL = CT + LABW + 0.08
CR = 7.08
CW = CR - CL
C_TOP = 2.28
CH = C_TOP - STRIP_B
axc = axes_in(CL, STRIP_B, CW, CH)
XLc = (0.645, 1.015)
YLc = (-0.55, 3.95)
axc.set_xlim(*XLc)
axc.set_ylim(*YLc)
SXc = CW / (XLc[1] - XLc[0])          # inches per x unit
SYc = CH / (YLc[1] - YLc[0])          # inches per y unit


def swarm_rows(vals, row, dot_in):
    """y positions so dots of diameter dot_in (inches) do not overlap; each dot as close to its row as it can."""
    placed, out = [], np.zeros(len(vals))
    for i in np.argsort(vals, kind="stable"):
        xi = vals[i] * SXc
        for k in range(400):
            cands = (0.0,) if k == 0 else (k * dot_in * 0.02, -k * dot_in * 0.02)
            ok = [yo for yo in cands if all((xi - a) ** 2 + (yo - b) ** 2 >= (dot_in * 1.06) ** 2 for a, b in placed)]
            if ok:
                break
        placed.append((xi, ok[0]))
        out[i] = row + ok[0] / SYc
    return out


axc.axvline(expert, color=MUTED, lw=0.6, zorder=1)
axc.text(expert - 0.012, 3.70, f"Pick expert {expert:.3f}", ha="right", va="center", color=INK2)
note("(c) Pick expert success (vertical line + label)", f"{expert} (printed 0.992)",
     "data/a/protect_np/ret/place_*.json and data/c/protect_c/ret/place_monitor_s*.json, key old_expert")
for arm, row, col, lab in C_ROWS:
    vals, srcs, key = ret[arm]
    mk = MARK[arm]
    ys_ = swarm_rows(vals, row, (MSIZE[mk] + 0.8) / 72)
    axc.plot(vals, ys_, mk, ms=MSIZE[mk], mfc=col, mec="white", mew=0.8, zorder=3, ls="none")
    axc.axhline(row, color=GUIDE, lw=0.5, zorder=0)          # row guide: label -> dots
    yin = STRIP_B + (row - YLc[0]) * SYc
    ftext(CL - 0.07, yin + 0.012, lab, ha="right", va="bottom")
    ftext(CL - 0.07, yin - 0.012, f"n={len(vals)}", ha="right", va="top", color=INK2)
    for v, s in zip(vals, srcs):
        note(f"(c) {lab} Pick success, final_ckpt, key {key}", f"{v}", rel(s))
    note(f"(c) {lab}: n printed", f"n={len(vals)}", "count of existing final-ckpt JSONs")
axc.set_yticks([])
axc.set_xticks([0.7, 0.8, 0.9, 1.0])
axc.set_xticklabels(["0.7", "0.8", "0.9", "1"])
for sp in ("top", "right", "left"):
    axc.spines[sp].set_visible(False)
ftext(CL + CW / 2, XLAB_Y, "Pick success rate", ha="center", va="baseline")
ftext(CT, TITLE_Y, "(c) Pick after Place", fontsize=8, fontweight="bold", va="baseline")
rich(CT, 2.405, [("Pick head reads the shared", 7, "normal", 0), 2.5, ("h", 7.5, "italic", 0),
                 ("′", 7.5, "normal", 0)], color=INK2)


# ================================================================ layout check
def layout_check():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for a in fig.axes:
        texts += a.texts
        if a.axison:
            texts += [a.xaxis.label, a.yaxis.label]
            texts += [tt for tt in a.get_xticklabels() + a.get_yticklabels() if tt.get_visible()]
    texts = [tt for tt in texts if tt.get_visible() and tt.get_text().strip()]
    tb = [(tt, tt.get_window_extent(r)) for tt in texts]
    W, H = fig.bbox.width, fig.bbox.height
    px_in = fig.dpi
    issues = []
    for tt, b in tb:
        if tt.get_fontsize() < 7 - 1e-9:
            issues.append(f"SMALL {tt.get_fontsize()} {tt.get_text()!r}")
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            issues.append(f"OFF-PAGE {tt.get_text()!r}")
    for (t1, b1), (t2, b2) in itertools.combinations(tb, 2):
        g1, g2 = RICH_GROUP.get(t1), RICH_GROUP.get(t2)
        if g1 is not None and g1 == g2:
            continue
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox > 0 and oy > 0:
            issues.append(f"TEXT/TEXT {t1.get_text()!r} | {t2.get_text()!r} ({ox / px_in:.3f} x {oy / px_in:.3f} in)")
    # markers: inside their axes, clear of every text, clear of each other within a series
    marks = []
    for a in fig.axes:
        for ln in a.lines:
            if ln.get_marker() in (None, "None", "", " "):
                continue
            rad = (ln.get_markersize() + ln.get_markeredgewidth()) / 2 / 72 * px_in
            for xy in ln.get_xydata():
                cx, cy = ln.get_transform().transform(xy)
                marks.append((a, cx, cy, rad))
    for a, cx, cy, rad in marks:
        bb = a.get_window_extent(r)
        if cx - rad < bb.x0 - 0.5 or cx + rad > bb.x1 + 0.5 or cy - rad < bb.y0 - 0.5 or cy + rad > bb.y1 + 0.5:
            issues.append(f"MARKER OUTSIDE AXES at ({cx / px_in:.2f}, {cy / px_in:.2f}) in")
        for tt, b in tb:
            if b.x0 < cx + rad and b.x1 > cx - rad and b.y0 < cy + rad and b.y1 > cy - rad:
                # circle vs box distance
                dx = max(b.x0 - cx, 0, cx - b.x1)
                dy = max(b.y0 - cy, 0, cy - b.y1)
                if dx * dx + dy * dy < rad * rad:
                    issues.append(f"TEXT/MARKER {tt.get_text()!r} at ({cx / px_in:.2f}, {cy / px_in:.2f}) in")
    return len(tb), len(marks), issues


for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(HERE, f"fig1.{ext}"), dpi=300)

n_text, n_mark, issues = layout_check()
for tt in fig.texts + [x for a in fig.axes for x in a.texts]:      # SVG would drop these spaces
    s_ = tt.get_text()
    if s_ and (s_[0] == " " or s_[-1] == " " or "  " in s_):
        issues.append(f"SVG-SPACE {s_!r}")
print(f"layout check: {n_text} texts, {n_mark} markers, {len(issues)} issues")
for i in issues:
    print("  ", i)

print("\nnumbers shown:")
for what, value, src in NUMBERS:
    print(f"  {what}: {value}  <- {src}")
print("\ncaption-only numbers:")
for arm in ("nsa64np", "nsa64npr"):
    print(f"  paired gaps vs ours ({arm}, eval intervals): {[int(g) for g in GAPS[arm]]}; exact one-sided signed-rank p = "
          f"{PVAL[arm]:.4f}")
print(f"  seed pairs where ours is faster: {N_FASTER}/{N_PAIRS}")
print(f"  StackCube median t90 (M): " + ", ".join(f"{a} {STACK_MED[a]:.4f}" for a in STACK_MED))
print(f"  exactness: stored {X_STORED_MIN:.1e}-{X_STORED_MAX:.1e}, random max {X_RANDOM_MAX:.1e}, "
      f"ours without P {X_NOPROJ[0]}-{X_NOPROJ[1]}, prefix-sum via h' {X_PS[0]}-{X_PS[1]}  <- {rel(EXACT)}")
for arm in ("nsa64", "nsa64np", "nsa64npr", "monitor"):
    v = ret[arm][0]
    print(f"  retention range {arm}: {v.min():.4f}-{v.max():.4f} (n={len(v)})")
