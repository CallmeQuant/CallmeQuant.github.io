"""Posterior consistency for a Bernoulli probability: contraction under a uniform prior, exponential
decay of the posterior mass outside a neighbourhood, and concentration at the Kullback-Leibler
projection when the prior support excludes the truth (note: posterior consistency)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from style import use_style, PALETTE, CYCLE, save

use_style()
theta0, eps = 0.3, 0.05

# ---- Figure 1: uniform prior, Beta(1 + S_n, 1 + n - S_n) posterior
rng = np.random.default_rng(3)
x = rng.random(10_000) < theta0
S = np.cumsum(x)
t = np.linspace(0, 1, 2001)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axes[0]
for k, n in enumerate([10, 100, 1000, 10000]):
    ax.plot(t, beta.pdf(t, 1 + S[n - 1], 1 + n - S[n - 1]), color=CYCLE[k], label=rf"$n = {n}$")
ax.axvline(theta0, color=PALETTE["grey"], lw=0.9, ls=":")
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 95)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title("Posterior along one sample path", loc="left")
ax.legend(loc="upper right")

ax = axes[1]
ns = np.arange(1, 3001)
for j in range(8):
    xs = rng.random(ns.size) < theta0
    Sj = np.cumsum(xs)
    a, b = 1 + Sj, 1 + ns - Sj
    out = beta.cdf(theta0 - eps, a, b) + beta.sf(theta0 + eps, a, b)
    ax.plot(ns, np.maximum(out, 1e-300), color=PALETTE["blue"], lw=1.0, alpha=0.75)
ax.set_yscale("log")
ax.set_xlim(0, 3000)
ax.set_ylim(1e-10, 3)
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"$\Pi_n(\vert\theta - 0.3\vert > 0.05)$")
ax.set_title("Mass outside a neighbourhood", loc="left")
save(fig, __file__, "contraction")

# ---- Figure 2: prior uniform on [0, 0.2] U [0.5, 1]
def kl(t0, t):
    return t0 * np.log(t0 / t) + (1 - t0) * np.log((1 - t0) / (1 - t))

tt = np.linspace(0.001, 0.999, 2000)
in_supp = (tt <= 0.2) | (tt >= 0.5)
rng = np.random.default_rng(99)
x = rng.random(400) < theta0
S = np.cumsum(x)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axes[0]
ax.axvspan(0, 0.2, color=PALETTE["light"], lw=0)
ax.axvspan(0.5, 1, color=PALETTE["light"], lw=0)
ax.plot(tt, kl(theta0, tt), color=PALETTE["ink"], lw=1.6)
ax.plot([0.2], [kl(theta0, 0.2)], "o", color=PALETTE["red"], ms=6)
ax.plot([0.5], [kl(theta0, 0.5)], "o", color=PALETTE["grey"], ms=6)
ax.annotate(r"$\theta^* = 0.2$", xy=(0.2, kl(theta0, 0.2)), xytext=(0.33, 0.18), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=PALETTE["ink"], lw=0.6))
ax.axvline(theta0, color=PALETTE["grey"], lw=0.9, ls=":")
ax.text(0.31, 0.62, r"$\theta_0$", fontsize=10)
ax.text(0.64, 0.62, "prior support", ha="center", fontsize=9.5, color=PALETTE["ink"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.7)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$K(\theta_0, \theta)$")
ax.set_title("Divergence from the truth", loc="left")
ax.grid(False)

ax = axes[1]
for k, n in enumerate([25, 100, 400]):
    s = S[n - 1]
    logpost = s * np.log(tt) + (n - s) * np.log1p(-tt)
    logpost[~in_supp] = -np.inf
    w = np.exp(logpost - logpost[in_supp].max())
    w /= np.trapezoid(w, tt)
    ax.plot(tt, w, color=CYCLE[k], label=rf"$n = {n}$")
ax.axvline(theta0, color=PALETTE["grey"], lw=0.9, ls=":")
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 165)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title(r"Posterior concentrates at $\theta^*$", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "misspecified-support")
