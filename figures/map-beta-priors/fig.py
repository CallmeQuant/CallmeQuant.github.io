"""Symmetric Beta priors for a coin's head probability and the posteriors after five heads in five
tosses, with posterior modes (note: maximum a posteriori estimation)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from style import use_style, PALETTE, CYCLE, save

use_style()
t = np.linspace(0.0005, 0.9995, 800)
alphas = [1, 2, 5, 10]
n1, n2 = 5, 0
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)

for k, a in enumerate(alphas):
    axes[0].plot(t, beta.pdf(t, a, a), color=CYCLE[k], label=rf"$\alpha=\beta={a}$")
axes[0].set_title(r"Priors $\mathrm{Beta}(\alpha, \alpha)$", loc="left")
axes[0].legend(loc="upper left")

for k, a in enumerate(alphas):
    pa, pb = n1 + a, n2 + a
    axes[1].plot(t, beta.pdf(t, pa, pb), color=CYCLE[k], label=rf"$\mathrm{{Beta}}({pa}, {pb})$")
    mode = (pa - 1) / (pa + pb - 2)
    axes[1].plot([mode], [beta.pdf(min(mode, 0.9995), pa, pb)], "o", color=CYCLE[k], ms=5, zorder=4, clip_on=False)
axes[1].axvline(1.0, color=PALETTE["grey"], lw=1.0, ls=(0, (4, 3)))
axes[1].text(0.985, 6.25, "MLE", ha="right", va="center", fontsize=9.5, color=PALETTE["grey"])
axes[1].set_title(r"Posteriors after 5 heads in 5 tosses; dots: MAP", loc="left")
axes[1].legend(loc="upper left")

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.6)
    ax.set_xlabel(r"$\theta$")
axes[0].set_ylabel("density")
save(fig, __file__, "beta-densities")
