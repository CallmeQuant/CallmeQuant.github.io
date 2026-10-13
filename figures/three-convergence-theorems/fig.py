"""Escaping mass and domination (note: three-convergence-theorems).
(a) f_n = n 1_(0,1/n]: mass escapes vertically; (b) f_n = 1_[n,n+1): mass escapes to infinity;
(c) f_n(x) = (1 + x/n)^(-n), dominated by (1 + x/2)^(-2), converges to exp(-x)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
ns = [1, 2, 4, 8]
cols = [PALETTE["gold"], PALETTE["green"], PALETTE["red"], PALETTE["blue"]]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))

ax = axes[0]
for n, c in zip(ns, cols):
    ax.plot([0, 0, 1 / n, 1 / n, 1.2], [0, n, n, 0, 0], color=c, lw=1.5, label=fr"$n = {n}$")
ax.set_xlim(-0.05, 1.2); ax.set_ylim(0, 9)
ax.set_xlabel(r"$x$")
ax.set_title(r"(a) $f_n = n\,\mathbf{1}_{(0, 1/n]}$", loc="left")
ax.legend(loc="upper right")

ax = axes[1]
for n, c in zip(ns, cols):
    ax.plot([0, n, n, n + 1, n + 1, 10], [0, 0, 1, 1, 0, 0], color=c, lw=1.5)
ax.set_xlim(0, 10); ax.set_ylim(0, 1.4)
ax.set_xlabel(r"$x$")
ax.set_title(r"(b) $f_n = \mathbf{1}_{[n, n+1)}$", loc="left")

ax = axes[2]
x = np.linspace(0, 6, 400)
ax.fill_between(x, 0, (1 + x / 2) ** -2, color=PALETTE["light"], lw=0, label=r"$g = (1 + x/2)^{-2}$")
for n, c in zip(ns[1:], cols[1:]):
    ax.plot(x, (1 + x / n) ** -n, color=c, lw=1.5)
ax.plot(x, np.exp(-x), color=PALETTE["ink"], ls="--", lw=1.2, label=r"$e^{-x}$")
ax.set_xlim(0, 6); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$x$")
ax.set_title(r"(c) $f_n = (1 + x/n)^{-n}$, $n \geq 2$", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "escaping-mass")
