"""Upper-tail bounds for S - np, S ~ Binomial(n, p) with n = 200 (note: moment-bounds).
Exact tail versus Chebyshev, Hoeffding, Bernstein and the optimized Chernoff bound, for p = 0.5
and p = 0.05."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
n = 200


def kl(q, p):
    q = np.clip(q, 1e-300, 1 - 1e-16)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
for ax, p, tmax in zip(axes, [0.5, 0.05], [45, 30]):
    t = np.linspace(0.5, tmax, 300)
    var = n * p * (1 - p)
    M = max(p, 1 - p)
    exact = stats.binom.sf(np.ceil(n * p + t) - 1, n, p)          # P(S >= np + t)
    ax.plot(t, exact, color=PALETTE["ink"], drawstyle="steps-post", label="exact")
    ax.plot(t, np.minimum(1, var / t**2), color=PALETTE["gold"], ls=":", label="Chebyshev")
    ax.plot(t, np.exp(-2 * t**2 / n), color=PALETTE["red"], ls="--", label="Hoeffding")
    ax.plot(t, np.exp(-t**2 / (2 * var + 2 * M * t / 3)), color=PALETTE["green"], ls="-.", label="Bernstein")
    ax.plot(t, np.exp(-n * kl(p + t / n, p)), color=PALETTE["blue"], label="Chernoff (optimized)")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.5)
    ax.set_xlabel(r"deviation $t$")
    ax.set_title(fr"$p = {p}$, $n p (1-p) = {var:g}$", loc="left")
axes[0].set_ylabel(r"$\mathbb{P}(S - np \geq t)$ and bounds")
axes[1].legend(loc="lower left")
save(fig, __file__, "binomial-tails")
