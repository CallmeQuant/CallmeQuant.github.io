"""Independent Bernoulli(p_n) indicators: times at which X_n = 1 (note: almost sure convergence
versus convergence in probability)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(20191111)
N, paths = 10_000, 10
n = np.arange(1, N + 1)
cases = [(r"$p_n = 1/n$: $\sum p_n = \infty$, ones occur infinitely often", 1 / n, PALETTE["blue"]),
         (r"$p_n = 1/n^2$: $\sum p_n < \infty$, ones eventually stop", 1 / n**2, PALETTE["red"])]

fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True)
for ax, (title, p, col) in zip(axes, cases):
    for j in range(paths):
        hits = n[rng.random(N) < p]
        ax.scatter(hits, np.full(hits.size, j + 1), marker="|", s=70, lw=1.4, color=col)
    ax.set_xscale("log")
    ax.set_xlim(0.8, N * 1.25)
    ax.set_ylim(0.3, paths + 0.7)
    ax.set_yticks([1, 5, 10])
    ax.set_ylabel("path")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
axes[1].set_xlabel(r"time $n$ (log scale)")
save(fig, __file__, "indicator-paths")
