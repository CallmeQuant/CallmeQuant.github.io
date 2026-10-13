"""Three copulas with Kendall's tau = 0.5, on the unit square and with fixed margins
(note: Sklar's theorem)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, kendalltau, pearsonr
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1959)
n, tau = 1500, 0.5
rho = np.sin(np.pi * tau / 2)          # Gaussian copula: tau = (2/pi) arcsin(rho)
theta = 2 * tau / (1 - tau)            # Clayton: tau = theta / (theta + 2)

z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
U_gauss = norm.cdf(z)
V = rng.gamma(1 / theta, 1.0, size=n)  # Marshall-Olkin sampler for Clayton
E = rng.exponential(size=(n, 2))
U_clay = (1 + E / V[:, None]) ** (-1 / theta)
U_surv = 1 - (1 + rng.exponential(size=(n, 2)) / rng.gamma(1 / theta, 1.0, size=n)[:, None]) ** (-1 / theta)

cases = [("Gaussian", U_gauss, PALETTE["blue"]),
         ("Clayton", U_clay, PALETTE["red"]),
         ("Survival Clayton", U_surv, PALETTE["green"])]

fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
for j, (name, U, col) in enumerate(cases):
    ax = axes[0, j]
    ax.scatter(U[:, 0], U[:, 1], s=2.5, color=col, alpha=0.6, lw=0, rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_title(f"{name} copula", loc="left")
    ax.set_xlabel(r"$u$")
    if j == 0:
        ax.set_ylabel(r"$v$")
    ax.text(0.03, 0.97, rf"$\hat\tau = {kendalltau(U[:, 0], U[:, 1])[0]:.2f}$", transform=ax.transAxes,
            va="top", fontsize=9.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    X, Y = expon.ppf(U[:, 0]), norm.ppf(U[:, 1])
    ax = axes[1, j]
    ax.scatter(X, Y, s=2.5, color=col, alpha=0.6, lw=0, rasterized=True)
    ax.set_xlim(0, 6.5); ax.set_ylim(-3.6, 3.6)
    ax.set_xlabel(r"$x = F_X^{-1}(u)$, Exp(1)")
    if j == 0:
        ax.set_ylabel(r"$y = \Phi^{-1}(v)$")
    ax.text(0.97, 0.04, rf"Pearson $r = {pearsonr(X, Y)[0]:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
save(fig, __file__, "three-copulas")
