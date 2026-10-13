"""Maximum entropy pmfs on {0,...,10} with a mean constraint (note: maximum entropy under moment constraints)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import binom
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2026)
K = 10
k = np.arange(K + 1)


def maxent(m):
    """pmf proportional to exp(theta k) on {0..K} with mean m (0 < m < K)."""
    def mean_gap(th):
        w = np.exp(th * k - np.max(th * k))
        return (w * k).sum() / w.sum() - m
    th = brentq(mean_gap, -60, 60)
    w = np.exp(th * k - np.max(th * k))
    return w / w.sum(), th


def H(p):
    p = np.asarray(p)
    q = np.where(p > 0, p, 1.0)
    return -(p * np.log(q)).sum(axis=-1)


fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
means = [1.0, 3.0, 5.0, 7.5]
width = 0.2
for j, m in enumerate(means):
    p, th = maxent(m)
    ax.bar(k + (j - 1.5) * width, p, width=width, color=CYCLE[j],
           label=rf"$m = {m:g}$, $\theta = {th:.2f}$")
ax.set_xticks(k)
ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$p_\theta(k)$")
ax.set_ylim(0, 0.68)
ax.set_title(r"Maximizers $p_\theta(k) \propto e^{\theta k}$", loc="left")
ax.legend(loc="upper center", ncol=2, fontsize=8.5, columnspacing=1.0)
ax.grid(axis="x", visible=False)

ax = axes[1]
P = rng.dirichlet(np.ones(K + 1), size=6000)
ax.scatter(P @ k, H(P), s=2, color=PALETTE["grey"], alpha=0.35, lw=0, rasterized=True,
           label="random pmfs (Dirichlet)")
B = np.array([binom.pmf(k, K, m / K) for m in np.linspace(0.05, K - 0.05, 200)])
ms = np.linspace(0.05, K - 0.05, 300)
ax.plot(ms, [H(maxent(m)[0]) for m in ms], color=PALETTE["blue"], lw=2, label="maximum entropy")
ax.plot(B @ k, H(B), color=PALETTE["red"], lw=1.4, ls="--", label=r"Binomial$(10, m/10)$")
ax.axhline(np.log(K + 1), color=PALETTE["ink"], lw=0.7, ls=":")
ax.text(0.2, np.log(K + 1) + 0.04, r"$\log 11$", fontsize=9.5, va="bottom")
ax.set_xlim(0, K); ax.set_ylim(0, 2.75)
ax.set_xlabel(r"mean $m$"); ax.set_ylabel("entropy (nats)")
ax.set_title(r"Entropy against mean", loc="left")
ax.legend(loc="lower center", fontsize=8.5, markerscale=4)
save(fig, __file__, "maxent-discrete")
