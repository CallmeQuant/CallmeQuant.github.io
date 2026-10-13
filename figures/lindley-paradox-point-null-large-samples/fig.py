"""Lindley's paradox: posterior probability of a point null at a fixed z-statistic, and the
Bayesian critical value of |z| as functions of n (tau = sigma, pi0 = 1/2)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
n = np.logspace(0, 7, 600)
r = n  # r = n tau^2 / sigma^2 with tau = sigma
zs = [(1.96, "0.05"), (2.576, "0.01"), (3.291, "0.001")]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
ax = axes[0]
for k, (z, p) in enumerate(zs):
    B = np.sqrt(1 + r) * np.exp(-0.5 * z**2 * r / (1 + r))
    ax.plot(n, B / (1 + B), color=CYCLE[k], label=rf"$z = {z:.2f}$ ($p = {p}$)")
    lb = 1 / (1 + np.exp(z**2 / 2))
    ax.axhline(lb, color=CYCLE[k], lw=0.9, ls=(0, (4, 3)))
ax.set_xscale("log")
ax.set_xlim(1, 1e7)
ax.set_ylim(0, 1)
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"$P(H_0 \mid X)$")
ax.set_title(r"Posterior probability of $H_0$ at fixed $z$", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1)

ax = axes[1]
c = np.sqrt((1 + 1 / r) * np.log1p(r))
ax.plot(n, c, color=PALETTE["blue"], label=r"Bayes: $P(H_0 \mid X) = 1/2$")
ax.axhline(1.96, color=PALETTE["red"], lw=1.4, ls=(0, (4, 3)), label=r"5% test")
ax.axhline(3.291, color=PALETTE["gold"], lw=1.4, ls=(0, (4, 3)), label=r"0.1% test")
ax.set_xscale("log")
ax.set_xlim(1, 1e7)
ax.set_ylim(0, 4.6)
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"critical value of $\vert z\vert$")
ax.set_title(r"Rejection thresholds", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1)
save(fig, __file__, "lindley")
