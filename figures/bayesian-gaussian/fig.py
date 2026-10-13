"""Conjugate Bayesian inference for the normal model (note: bayesian-gaussian).
Left: posterior of the mean with known variance as n grows.
Right: marginal posterior of the mean under the normal-inverse-gamma prior (Student t)
versus a normal density with the same location and scale."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(2023)
mu_true, sigma = 1.0, 2.0
x = rng.normal(mu_true, sigma, 50)

# Known variance: prior N(m, tau^2)
m, tau = 0.0, 1.5
grid = np.linspace(-3, 4, 600)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
ax = axes[0]
ax.plot(grid, stats.norm.pdf(grid, m, tau), color=PALETTE["grey"], ls="--", label="prior")
for n, col in zip([2, 10, 50], [PALETTE["gold"], PALETTE["red"], PALETTE["blue"]]):
    prec = 1 / tau**2 + n / sigma**2
    mn = (m / tau**2 + x[:n].sum() / sigma**2) / prec
    ax.plot(grid, stats.norm.pdf(grid, mn, prec**-0.5), color=col, label=fr"$n = {n}$")
ax.axvline(mu_true, color=PALETTE["ink"], lw=0.8, ls=":")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("density")
ax.set_title(r"(a) known $\sigma^2$: posterior of $\mu$", loc="left")
ax.legend(loc="upper left")

# Unknown mean and variance: NIG(m0, kappa0, a0, b0) prior, n = 5 observations
m0, k0, a0, b0 = 0.0, 1.0, 1.0, 1.0
n = 5
xs = x[:n]; xb = xs.mean()
kn = k0 + n; mn = (k0 * m0 + n * xb) / kn
an = a0 + n / 2
bn = b0 + 0.5 * ((xs - xb) ** 2).sum() + k0 * n * (xb - m0) ** 2 / (2 * kn)
scale = np.sqrt(bn / (an * kn))
g2 = np.linspace(mn - 8 * scale, mn + 8 * scale, 600)
ax = axes[1]
ax.plot(g2, stats.t.pdf(g2, 2 * an, mn, scale), color=PALETTE["blue"],
        label=fr"Student $t$, ${2 * an:.0f}$ d.f.")
ax.plot(g2, stats.norm.pdf(g2, mn, scale), color=PALETTE["red"], ls="--",
        label="normal, same scale")
ax.set_yscale("log")
ax.set_ylim(1e-5, 2)
ax.set_xlabel(r"$\mu$")
ax.set_title(r"(b) unknown $\sigma^2$, $n = 5$ (log scale)", loc="left")
ax.legend(loc="lower center")
save(fig, __file__, "posteriors")
