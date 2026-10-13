"""Stopping rules and the likelihood principle: three successes in twelve Bernoulli trials under a
fixed-n (binomial) design and a sample-until-third-success (negative binomial) design, and the
common posterior under a uniform prior (note: objections to Bayesian inference)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
from scipy.special import comb
from style import use_style, PALETTE, save

use_style()
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))

ax = axes[0]
k = np.arange(0, 13)
pk = binom.pmf(k, 12, 0.5)
ax.bar(k, pk, color=np.where(k <= 3, PALETTE["red"], PALETTE["light"]), width=0.75)
ax.set_xlabel(r"successes $X$ in $n = 12$")
ax.set_ylabel("probability under $\\theta = 1/2$")
ax.set_title(r"Binomial: $P(X \leq 3) = 0.073$", loc="left", fontsize=10.5)
ax.set_xticks([0, 3, 6, 9, 12])
ax.set_ylim(0, 0.25)
ax.grid(axis="x", visible=False)

ax = axes[1]
n = np.arange(3, 26)
pn = comb(n - 1, 2) * 0.5 ** n          # P(third success on trial n)
ax.bar(n, pn, color=np.where(n >= 12, PALETTE["red"], PALETTE["light"]), width=0.75)
ax.set_xlabel(r"trial $N$ of the third success")
ax.set_title(r"Neg. binomial: $P(N \geq 12) = 0.033$", loc="left", fontsize=10.5)
ax.set_xticks([3, 6, 9, 12, 15, 18, 21, 24])
ax.set_ylim(0, 0.25)
ax.grid(axis="x", visible=False)

ax = axes[2]
t = np.linspace(0, 1, 600)
d = beta.pdf(t, 4, 10)
ax.plot(t, d, color=PALETTE["blue"])
ax.fill_between(t[t >= 0.5], d[t >= 0.5], color=PALETTE["red"], lw=0)
ax.annotate(r"$P(\theta \geq 1/2) = 0.046$", xy=(0.53, 0.25), xytext=(0.5, 1.6), fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=PALETTE["ink"], lw=0.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 3.6)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title(r"Posterior: Beta$(4, 10)$", loc="left", fontsize=10.5)
save(fig, __file__, "stopping-rule")
