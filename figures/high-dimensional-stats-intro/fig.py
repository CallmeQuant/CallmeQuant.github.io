"""(a) Tail bounds for a standard normal variable: Markov (applied to |X|), Chebyshev and Chernoff
against the exact two-sided tail. (b) Johnson-Lindenstrauss: squared-distance ratios after a Gaussian
random projection of N = 100 points from R^5000 to R^k (note: high-dimensional-stats-intro)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1916)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})

ax = axes[0]
a = np.linspace(0.5, 5, 400)
ax.plot(a, 2 * stats.norm.sf(a), color=PALETTE["ink"], label=r"exact $\mathbb{P}(|X| \geq a)$")
ax.plot(a, np.sqrt(2 / np.pi) / a, color=PALETTE["gold"], ls=":", label=r"Markov $\mathbb{E}|X|/a$")
ax.plot(a, 1 / a**2, color=PALETTE["red"], ls="--", label=r"Chebyshev $1/a^2$")
ax.plot(a, 2 * np.exp(-a**2 / 2), color=PALETTE["blue"], label=r"Chernoff $2e^{-a^2/2}$")
ax.set_yscale("log")
ax.set_ylim(1e-7, 2)
ax.set_xlabel(r"$a$")
ax.set_title(r"(a) tails of $X \sim \mathcal{N}(0,1)$", loc="left")
ax.legend(loc="lower left")

ax = axes[1]
N, d = 100, 5000
P = rng.normal(size=(N, d)) * rng.uniform(0.5, 2.0, size=(N, 1))   # points at varied scales
iu = np.triu_indices(N, 1)
def sqdist(Y):
    G = Y @ Y.T
    s = np.diag(G)
    return (s[:, None] + s[None, :] - 2 * G)[iu]
D0 = sqdist(P)
bins = np.linspace(0.2, 2.0, 121)
for k, col in zip([25, 100, 400], [PALETTE["gold"], PALETTE["red"], PALETTE["blue"]]):
    A = rng.normal(size=(k, d)) / np.sqrt(k)
    ratio = sqdist(P @ A.T) / D0
    ax.hist(ratio, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=col)
    ax.hist(ratio, bins=bins, density=True, histtype="step", lw=1.2, color=col, label=fr"$k = {k}$")
ax.axvline(1, color=PALETTE["ink"], lw=0.8, ls=":")
ax.set_xlabel(r"$\Vert f(x_i) - f(x_j)\Vert^2 / \Vert x_i - x_j\Vert^2$")
ax.set_ylabel("density")
ax.set_title(r"(b) random projection, $d = 5000$, $N = 100$", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "tails-and-projection")
