"""Gibbs posteriors for the median under absolute-error loss with Gamma(2, 1) data: posteriors for
several learning rates against a Gaussian-likelihood posterior, and frequentist coverage of 95%
credible intervals as a function of the learning rate (note: Gibbs posteriors)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from style import use_style, PALETTE, CYCLE, save

use_style()
shape = 2.0
med = gamma.ppf(0.5, shape)                 # target: population median, about 1.678
omega_star = 2 * gamma.pdf(med, shape)       # calibrated learning rate 2 f(m), about 0.627
grid = np.linspace(0.0, 8.0, 3201)           # uniform prior on [0, 8]
dg = grid[1] - grid[0]


def gibbs(x, omegas):
    """Gibbs posterior densities on the grid, one row per learning rate."""
    loss = np.abs(x[:, None] - grid[None, :]).sum(0)
    lw = -np.outer(omegas, loss)
    w = np.exp(lw - lw.max(1, keepdims=True))
    return w / (w.sum(1, keepdims=True) * dg)


# ---- one data set
rng = np.random.default_rng(31)
n = 50
x = rng.gamma(shape, 1.0, n)
omegas = np.array([0.2, omega_star, 3.0])
post = gibbs(x, omegas)
gauss = norm.pdf(grid, x.mean(), x.std(ddof=1) / np.sqrt(n))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
labels = [r"$\omega = 0.2$", rf"$\omega = \omega^* \approx {omega_star:.2f}$", r"$\omega = 3$"]
for k in range(3):
    ax.plot(grid, post[k], color=CYCLE[k], label=labels[k])
ax.plot(grid, gauss, color=PALETTE["grey"], ls=(0, (4, 3)), lw=1.5, label="Gaussian model")
ax.axvline(med, color=PALETTE["ink"], lw=0.8, ls=":")
ax.axvline(shape, color=PALETTE["grey"], lw=0.8, ls=":")
ytop = 1.12 * post.max()
ax.text(med - 0.03, 0.95 * ytop, "median", ha="right", fontsize=9)
ax.text(shape + 0.03, 0.95 * ytop, "mean", ha="left", fontsize=9, color=PALETTE["grey"])
ax.set_xlim(0.0, 3.4)
ax.set_ylim(0, ytop)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title(r"One sample, $n = 50$", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=8.8)

# ---- coverage of equal-tailed 95% intervals
reps = 2000
omega_grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 31))
ax = axes[1]
rng = np.random.default_rng(32)
for k, n in enumerate([50, 400]):
    hits = np.zeros(omega_grid.size)
    for _ in range(reps):
        x = rng.gamma(shape, 1.0, n)
        p = gibbs(x, omega_grid)
        cdf = np.cumsum(p, 1) * dg
        lo = grid[np.argmax(cdf >= 0.025, axis=1)]
        hi = grid[np.argmax(cdf >= 0.975, axis=1)]
        hits += (lo <= med) & (med <= hi)
    ax.plot(omega_grid, hits / reps, "o-", color=CYCLE[k], ms=3.2, lw=1.4, label=rf"$n = {n}$")
og = np.exp(np.linspace(np.log(0.05), np.log(5.0), 400))
ax.plot(og, 2 * norm.cdf(norm.ppf(0.975) * np.sqrt(omega_star / og)) - 1, color=PALETTE["ink"], lw=0.9,
        label="asymptotic")
ax.axhline(0.95, color=PALETTE["grey"], lw=0.9, ls=(0, (4, 3)))
ax.axvline(omega_star, color=PALETTE["ink"], lw=0.8, ls=":")
ax.text(omega_star * 1.08, 0.08, r"$\omega^*$", fontsize=10)
ax.set_xscale("log")
ax.set_xlim(0.05, 5)
ax.set_ylim(0, 1.02)
ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
ax.set_xlabel(r"learning rate $\omega$ (log scale)")
ax.set_ylabel("coverage of 95% interval")
ax.set_title("Frequentist coverage for the median", loc="left")
ax.legend(loc="lower left")
save(fig, __file__, "gibbs-median")
