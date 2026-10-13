"""Dirichlet process DP(alpha, N(0,1)): random distribution functions for three concentration
parameters (stick-breaking), stick-breaking weights, and the number of distinct values in a Polya
urn sequence (note: Dirichlet process)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()


def stick_breaking(rng, alpha, tol=1e-10):
    """Weights and atoms of a DP(alpha, N(0,1)) draw, truncated when the remaining stick < tol."""
    w, atoms, rest = [], [], 1.0
    while rest > tol:
        v = rng.beta(1.0, alpha)
        w.append(rest * v)
        atoms.append(rng.standard_normal())
        rest *= 1.0 - v
    return np.array(w), np.array(atoms)


# ---- Figure 1: random CDFs
rng = np.random.default_rng(1973)
x = np.linspace(-3, 3, 1201)
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), sharey=True)
for ax, alpha in zip(axes, [1, 10, 100]):
    for j in range(8):
        w, a = stick_breaking(rng, alpha)
        o = np.argsort(a)
        ax.step(np.concatenate([[-3], a[o], [3]]), np.concatenate([[0], np.cumsum(w[o]), [np.sum(w)]]),
                where="post", color=PALETTE["blue"], lw=0.9, alpha=0.7)
    ax.plot(x, norm.cdf(x), color=PALETTE["red"], lw=1.6, ls=(0, (4, 3)))
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$")
    ax.set_title(rf"$\alpha = {alpha}$", loc="left")
axes[0].set_ylabel(r"$P((-\infty, x])$")
save(fig, __file__, "dp-cdfs")

# ---- Figure 2: stick-breaking weights and the number of clusters
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axes[0]
rng = np.random.default_rng(1994)
K = 25
for k, (alpha, off) in enumerate([(2, -0.2), (20, 0.2)]):
    w, _ = stick_breaking(rng, alpha)
    ax.bar(np.arange(1, K + 1) + off, w[:K], width=0.38, color=CYCLE[k], label=rf"$\alpha = {alpha}$")
    ax.plot(np.arange(1, K + 1), (1 / (1 + alpha)) * (alpha / (1 + alpha)) ** np.arange(K),
            color=CYCLE[k], lw=1.0, ls=(0, (3, 2)))
ax.set_xlim(0.3, K + 0.7)
ax.set_ylim(0, 0.6)
ax.set_xlabel(r"index $k$")
ax.set_ylabel(r"weight $w_k$")
ax.set_title("Stick-breaking weights", loc="left")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)

ax = axes[1]
rng = np.random.default_rng(1953)
N = 5000
n = np.arange(1, N + 1)
for k, alpha in enumerate([1, 5, 25]):
    for _ in range(5):
        new = rng.random(N) < alpha / (alpha + n - 1)   # independent "new value" indicators
        ax.plot(n, np.cumsum(new), color=CYCLE[k], lw=0.8, alpha=0.55)
    ax.plot(n, np.cumsum(alpha / (alpha + n - 1)), color=CYCLE[k], lw=1.8, label=rf"$\alpha = {alpha}$")
ax.set_xscale("log")
ax.set_xlim(1, N)
ax.set_ylim(0, 180)
ax.set_xlabel(r"sample size $n$ (log scale)")
ax.set_ylabel(r"distinct values $K_n$")
ax.set_title("Number of distinct values", loc="left")
ax.legend(loc="upper left")
save(fig, __file__, "dp-weights-clusters")
