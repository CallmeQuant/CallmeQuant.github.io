"""Linear mixed model as a hierarchical model: partial pooling of group means by the BLUP in a
random-intercept model, and the sampling distributions of the ML and REML estimators of the
random-effect variance in a balanced one-way layout (note: mixed models and hierarchical models)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()

# ---- (a) unbalanced random-intercept model with known variance components
rng = np.random.default_rng(1974)
mu, tau, sigma = 10.0, 1.0, 2.0
nj = np.array([2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40])
J = nj.size
b = tau * rng.standard_normal(J)
ybar = np.array([mu + b[j] + sigma * rng.standard_normal(nj[j]).mean() for j in range(J)])
w = 1.0 / (tau**2 + sigma**2 / nj)
mu_hat = np.sum(w * ybar) / np.sum(w)                    # GLS estimate of mu
lam = nj * tau**2 / (nj * tau**2 + sigma**2)             # shrinkage factors
blup = mu_hat + lam * (ybar - mu_hat)

# ---- (b) balanced one-way layout: ML and REML estimators of tau^2
Jb, nb, tau2b, sig2b, reps = 6, 5, 1.0, 4.0, 20000
rng = np.random.default_rng(1977)
bb = np.sqrt(tau2b) * rng.standard_normal((reps, Jb, 1))
yb = 0.0 + bb + np.sqrt(sig2b) * rng.standard_normal((reps, Jb, nb))
gm = yb.mean((1, 2))
gmeans = yb.mean(2)
msb = nb * ((gmeans - gm[:, None]) ** 2).sum(1) / (Jb - 1)
msw = ((yb - gmeans[:, :, None]) ** 2).sum((1, 2)) / (Jb * (nb - 1))
reml = np.maximum((msb - msw) / nb, 0)
ml = np.maximum(((1 - 1 / Jb) * msb - msw) / nb, 0)
print("mean REML %.3f, mean ML %.3f, P(REML=0) %.3f, P(ML=0) %.3f"
      % (reml.mean(), ml.mean(), (reml == 0).mean(), (ml == 0).mean()))

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
ax.axhline(mu_hat, color=PALETTE["grey"], lw=0.9, ls=(0, (4, 3)))
for j in range(J):
    ax.annotate("", xy=(nj[j], blup[j]), xytext=(nj[j], ybar[j]),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["grey"], lw=0.8, shrinkA=3, shrinkB=3,
                                mutation_scale=7))
ax.scatter(nj, ybar, s=26, facecolor="white", edgecolor=PALETTE["ink"], lw=0.9, zorder=3, label=r"group mean $\bar y_j$")
ax.scatter(nj, blup, s=24, color=PALETTE["blue"], zorder=4, label=r"BLUP $\hat\mu + \hat b_j$")
ax.scatter(nj, mu + b, s=30, marker="x", color=PALETTE["red"], lw=1.1, zorder=4, label=r"true $\mu + b_j$")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 40])
ax.set_xticklabels(["2", "5", "10", "20", "40"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlim(1.7, 48)
ax.set_xlabel(r"group size $n_j$ (log scale)")
ax.set_ylabel("estimate of group mean")
ax.set_title("Partial pooling of group means", loc="left")
ax.legend(loc="lower right", fontsize=9)
lo, hi = min(ybar.min(), (mu + b).min()), max(ybar.max(), (mu + b).max())
ax.set_ylim(lo - 1.6, hi + 0.5)

ax = axes[1]
grid = np.linspace(0, 6, 1201)
for est, col, name in [(reml, PALETTE["blue"], "REML"), (ml, PALETTE["red"], "ML")]:
    ecdf = np.searchsorted(np.sort(est), grid, side="right") / est.size
    ax.plot(grid, ecdf, color=col, lw=1.6, label=rf"{name}: mean {est.mean():.2f}, $P(\hat\tau^2 = 0) = {np.mean(est == 0):.2f}$")
    ax.axvline(est.mean(), color=col, lw=1.0, ls=(0, (4, 3)))
ax.axvline(tau2b, color=PALETTE["ink"], lw=1.0)
ax.text(tau2b + 0.12, 0.93, r"true $\tau^2 = 1$", fontsize=9.5)
ax.set_xlim(0, 6)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"estimate of $\tau^2$")
ax.set_ylabel("distribution function")
ax.set_title(r"ML and REML for $\tau^2$, $J = 6$, $n = 5$", loc="left")
ax.legend(loc="lower right", fontsize=8.6)
save(fig, __file__, "pooling-reml")
