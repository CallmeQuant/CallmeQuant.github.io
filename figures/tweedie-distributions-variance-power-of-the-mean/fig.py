"""Tweedie distributions with 1 < p < 2 as compound Poisson-gamma laws (note: Tweedie distributions).
Left: continuous part of the density for mu = 1, phi = 1 and several p, with the mass at zero.
Right: sample variance against sample mean on log-log axes; slopes recover the power p."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1984)


def cpg_params(mu, phi, p):
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    gam = phi * (p - 1) * mu ** (p - 1)
    return lam, alpha, gam


def density(y, mu, phi, p, nmax=400):
    lam, alpha, gam = cpg_params(mu, phi, p)
    n = np.arange(1, nmax)[:, None]
    return np.sum(stats.poisson.pmf(n, lam) * stats.gamma.pdf(y[None, :], n * alpha, scale=gam), axis=0)


def rtweedie(size, mu, phi, p):
    lam, alpha, gam = cpg_params(mu, phi, p)
    N = rng.poisson(lam, size)
    return np.where(N > 0, rng.gamma(np.maximum(N, 1) * alpha, gam), 0.0)


fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
y = np.linspace(0.005, 4, 800)
for col, p in zip(CYCLE, [1.1, 1.3, 1.5, 1.7, 1.9]):
    lam, _, _ = cpg_params(1.0, 1.0, p)
    p0 = np.exp(-lam)
    p0s = f"{p0:.2f}" if p0 >= 0.01 else r"{:.1f} \times 10^{{{}}}".format(*map(float, f"{p0:.1e}".split("e")[0:1]), int(f"{p0:.1e}".split("e")[1]))
    ax.plot(y, density(y, 1.0, 1.0, p), color=col, lw=1.5, label=rf"$p = {p}$:  $\mathrm{{P}}(Y=0) = {p0s}$")
ax.set_xlim(0, 4); ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$y$"); ax.set_ylabel("density of the continuous part")
ax.set_title(r"$\mu = 1$, $\phi = 1$", loc="left")
ax.legend(loc="upper right", fontsize=8.6)

ax = axes[1]
mus = np.logspace(-1, 1, 9)
for col, p in zip([CYCLE[0], CYCLE[2], CYCLE[4]], [1.2, 1.5, 1.8]):
    v = [rtweedie(4000, m, 1.0, p).var(ddof=1) for m in mus]
    ax.loglog(mus, v, "o", color=col, mfc="white", mew=1.3, ms=5)
    ax.loglog(mus, mus**p, color=col, lw=1.2, label=rf"$\mathrm{{Var}}\,Y = \mu^{{{p}}}$")
ax.set_xlabel(r"mean $\mu$ (log scale)"); ax.set_ylabel(r"variance (log scale)")
ax.set_title(r"Sample variances, $n = 4000$ per point", loc="left")
ax.legend(loc="upper left", fontsize=8.8)
save(fig, __file__, "tweedie-cpg")
