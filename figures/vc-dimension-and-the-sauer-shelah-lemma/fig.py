"""VC dimension (note: VC dimension and the Sauer-Shelah lemma).
Figure 1: three points in the plane shattered by half-planes; four points cannot be (Radon partitions).
Figure 2: growth functions of three classes against 2^n and the Sauer-Shelah bound."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.special import comb
from style import use_style, PALETTE, save

use_style()
POS, NEG = PALETTE["blue"], PALETTE["red"]


def separating_line(P, y):
    """A half-plane {x : w.x + b > 0} containing exactly the points with y = 1 (simple construction for 3 points)."""
    if y.all() or (~y).all():
        return None
    odd = y if y.sum() == 1 else ~y                # the singleton class
    s = P[odd][0]; others = P[~odd]
    mid = others.mean(axis=0)
    m = 0.5 * (s + mid)                             # passes between the singleton and the other two
    d = others[1] - others[0]
    w = np.array([-d[1], d[0]])                     # normal parallel to the segment's normal direction
    if np.dot(w, s - m) < 0:
        w = -w
    if y.sum() != 1:                                # positive side should contain the positive points
        w = -w
    return w, -np.dot(w, m)


def draw_halfplane(ax, w, b, lim=(-0.6, 1.6)):
    xs = np.linspace(*lim, 2)
    if abs(w[1]) > 1e-9:
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color=PALETTE["ink"], lw=1.0)
        top = lim[1] + 5 if w[1] > 0 else lim[0] - 5
        ax.add_patch(Polygon([[xs[0], ys[0]], [xs[1], ys[1]], [xs[1], top], [xs[0], top]],
                             closed=True, color=POS, alpha=0.10, lw=0))


P3 = np.array([[0.0, 0.0], [1.0, 0.15], [0.35, 1.0]])
fig = plt.figure(figsize=(7.8, 4.9), layout=None)


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(PALETTE["light"])


fig.text(0.01, 0.985, "Three points: all $2^3 = 8$ labelings are realized by half-planes (shaded side = label 1)",
         ha="left", va="top", fontsize=10.5)
W, H, GAP = 0.235, 0.17, 0.012
for k, lab in enumerate(itertools.product([0, 1], repeat=3)):
    y = np.array(lab, bool)
    r, c = divmod(k, 4)
    ax = fig.add_axes([0.01 + c * (W + GAP), 0.75 - r * (H + GAP), W, H])
    sl = separating_line(P3, y)
    if sl is not None:
        draw_halfplane(ax, *sl)
    elif y.all():
        ax.set_facecolor((0.169, 0.298, 0.608, 0.10))
    ax.scatter(P3[y, 0], P3[y, 1], color=POS, s=26, zorder=3)
    ax.scatter(P3[~y, 0], P3[~y, 1], facecolor="white", edgecolor=NEG, s=26, lw=1.4, zorder=3)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.3, 1.3)
    frame(ax)

fig.text(0.01, 0.47, "Four points: the Radon partition (filled vs open) is never realized, because the convex hulls meet",
         ha="left", va="top", fontsize=10.5)
cases = [(np.array([[0, 0], [1.2, 0.1], [1.1, 1.0], [0.1, 0.9]]), [0, 2], "convex position: the diagonals cross"),
         (np.array([[0, 0], [1.3, 0.1], [0.6, 1.1], [0.62, 0.42]]), [3], "one point inside the others' triangle")]
for c, (P, I, title) in enumerate(cases):
    ax = fig.add_axes([0.01 + c * 0.495, 0.0, 0.485, 0.40])
    J = [j for j in range(4) if j not in I]
    for S, col in [(I, POS), (J, NEG)]:
        Q = P[S]
        if len(Q) == 2:
            ax.plot(Q[:, 0], Q[:, 1], color=col, lw=1.4)
        elif len(Q) >= 3:
            ax.add_patch(Polygon(Q, closed=True, facecolor=mpl.colors.to_rgba(col, 0.12), edgecolor=col, lw=1.2))
    ax.scatter(P[I, 0], P[I, 1], color=POS, s=34, zorder=3)
    ax.scatter(P[J, 0], P[J, 1], facecolor="white", edgecolor=NEG, s=34, lw=1.4, zorder=3)
    ax.set_xlim(-0.3, 1.6); ax.set_ylim(-0.25, 1.3)
    frame(ax)
    ax.text(0.02, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=PALETTE["ink"])
save(fig, __file__, "shattering")

# ---- Figure 2 ----
n = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.semilogy(n, 2.0**n, color=PALETTE["ink"], lw=1.2, ls="--", label=r"$2^n$")
for d, f, col, lab in [(1, n + 1, PALETTE["blue"], "thresholds on $\\mathbb{R}$ ($d = 1$)"),
                       (2, comb(n, 2) + n + 1, PALETTE["green"], r"intervals on $\mathbb{R}$ ($d = 2$)"),
                       (3, n * n - n + 2, PALETTE["red"], r"half-planes in $\mathbb{R}^2$ ($d = 3$)")]:
    ax.semilogy(n, f, "o", color=col, ms=3.5, label=lab)
    sauer = sum(comb(n, i) for i in range(d + 1))
    ax.semilogy(n, sauer, color=col, lw=1.0)
ax.set_xlim(0, 31); ax.set_ylim(1, 1e5)
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"growth function $\Pi_{\mathcal{H}}(n)$")
ax.legend(loc="upper left", fontsize=8.6)
ax.text(30.5, 2.0, r"lines: Sauer–Shelah bound $\sum_{i \leq d}\binom{n}{i}$", ha="right", fontsize=9.5, color=PALETTE["ink"])
save(fig, __file__, "growth-functions")
