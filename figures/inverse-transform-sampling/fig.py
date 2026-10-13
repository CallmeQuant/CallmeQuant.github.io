"""Generalized inverse of a distribution function with a flat stretch and a jump
(note: inverse transform sampling). Mixture: 0.35 Unif(0,1) + 0.25 delta_{1.5} + 0.40 Unif(2,3)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save


def F(x):
    x = np.asarray(x, float)
    return np.where(x < 0, 0.0,
           np.where(x < 1, 0.35 * x,
           np.where(x < 1.5, 0.35,
           np.where(x < 2, 0.60,
           np.where(x < 3, 0.60 + 0.40 * (x - 2), 1.0)))))


def Finv(u):
    """Generalized inverse inf{x : F(x) >= u} for u in (0, 1)."""
    u = np.asarray(u, float)
    return np.where(u <= 0.35, u / 0.35, np.where(u <= 0.60, 1.5, 2 + (u - 0.60) / 0.40))


use_style()
ink, blue, red, grey = PALETTE["ink"], PALETTE["blue"], PALETTE["red"], PALETTE["grey"]
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))

# (a) F with three uniforms mapped to x
ax = axes[0]
segs = [(-0.4, 0, 0, 0), (0, 1, 0, 0.35), (1, 1.5, 0.35, 0.35), (1.5, 2, 0.6, 0.6), (2, 3, 0.6, 1.0), (3, 3.4, 1, 1)]
for x0, x1, y0, y1 in segs:
    ax.plot([x0, x1], [y0, y1], color=blue)
ax.plot([1.5], [0.35], "o", mfc="white", mec=blue, ms=5, zorder=4)
ax.plot([1.5], [0.6], "o", color=blue, ms=5, zorder=4)
for u, col in [(0.2, red), (0.5, red), (0.8, red)]:
    x = float(Finv(u))
    ax.annotate("", xy=(x, 0), xytext=(x, u), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    ax.plot([-0.4, x], [u, u], color=col, lw=1.0, ls=(0, (3, 2)))
ax.set_xlim(-0.4, 3.4); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$x$"); ax.set_title(r"(a) distribution function $F$", loc="left")
ax.set_yticks([0, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0", "0.2", "0.35", "0.5", "0.6", "0.8", "1"])

# (b) generalized inverse
ax = axes[1]
for u0, u1 in [(0, 0.35), (0.35, 0.6), (0.6, 1.0)]:
    uu = np.linspace(u0, u1, 50)
    v = Finv(uu)
    if u0 > 0:  # left-continuous: value at u0 belongs to previous piece
        v[0] = float(Finv(u0 + 1e-12))
    ax.plot(uu, v, color=blue)
ax.plot([0.35, 0.6], [1.0, 1.5], "o", color=blue, ms=5, zorder=4)
ax.plot([0.35, 0.6], [1.5, 2.0], "o", mfc="white", mec=blue, ms=5, zorder=4)
ax.set_xlim(0, 1); ax.set_ylim(-0.1, 3.2)
ax.set_xlabel(r"$u$"); ax.set_title(r"(b) quantile function $F^{-1}$", loc="left")
ax.set_xticks([0, 0.35, 0.6, 1.0]); ax.set_xticklabels(["0", "0.35", "0.6", "1"])

# (c) empirical distribution function of inverse-transform draws
rng = np.random.default_rng(2023)
X = np.sort(Finv(rng.random(400)))
ax = axes[2]
xx = np.linspace(-0.4, 3.4, 1000)
ax.plot(xx, F(xx), color=grey, lw=3.2, alpha=0.6, label=r"$F$")
ax.step(np.r_[-0.4, X, 3.4], np.r_[0, np.arange(1, X.size + 1) / X.size, 1], where="post",
        color=red, lw=1.0, label="empirical, $n = 400$")
ax.set_xlim(-0.4, 3.4); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$x$"); ax.set_title(r"(c) draws $F^{-1}(U_i)$", loc="left")
ax.legend(loc="upper left")
save(fig, __file__, "generalized-inverse")
print("fraction of draws at 1.5:", np.mean(X == 1.5))
