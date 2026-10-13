"""Lasso sign recovery and the irrepresentable condition (note: when the lasso finds the right variables).
Design: X1, X2 independent N(0,1); X3 has correlation c with each of X1, X2; beta* = (1, 1, 0).
The irrepresentable quantity is 2c, so the condition holds for c = 0.4 and fails for c = 0.65."""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, lasso_path
from style import use_style, PALETTE, save

use_style()
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2006)
SIGMA, B = 1.0, 1.0
cases = [(0.4, r"$c = 0.4$: $\Vert \Gamma z_S \Vert_\infty = 0.8$"),
         (0.65, r"$c = 0.65$: $\Vert \Gamma z_S \Vert_\infty = 1.3$")]


def design(n, c, p_extra=0):
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    x3 = c * x1 + c * x2 + np.sqrt(1 - 2 * c * c) * rng.standard_normal(n)
    X = np.column_stack([x1, x2, x3] + [rng.standard_normal(n) for _ in range(p_extra)])
    y = B * (x1 + x2) + SIGMA * rng.standard_normal(n)
    return X, y


# ---- Figure 1: lasso paths, p = 3 ----
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharey=True)
lams = np.logspace(0.3, -3, 200)
for ax, (c, title) in zip(axes, cases):
    X, y = design(1000, c)
    _, coefs, _ = lasso_path(X, y, alphas=lams)
    ax.plot(lams, coefs[0], color=PALETTE["blue"], label=r"$\hat\beta_1$")
    ax.plot(lams, coefs[1], color=PALETTE["green"], ls="--", label=r"$\hat\beta_2$")
    ax.plot(lams, coefs[2], color=PALETTE["red"], label=r"$\hat\beta_3$ (true value 0)")
    ax.axhline(0, color=PALETTE["ink"], lw=0.6)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda$ (log scale, decreasing)")
    ax.set_title(title, loc="left", fontsize=10.5)
axes[0].set_ylabel("lasso coefficient")
axes[0].legend(loc="center right", fontsize=8.8)
save(fig, __file__, "lasso-paths")

# ---- Figure 2: probability of exact sign recovery, p = 50 ----
ns = [100, 200, 400, 800, 1600, 3200, 6400]
R, p = 200, 50
truth = np.sign(np.r_[B, B, np.zeros(p - 2)])
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for (c, title), col, mk in zip(cases, [PALETTE["blue"], PALETTE["red"]], ["o", "s"]):
    prob = []
    for n in ns:
        lam = 4 * SIGMA * np.sqrt(2 * np.log(p) / n)
        hits = 0
        for _ in range(R):
            X, y = design(n, c, p - 3)
            b = Lasso(alpha=lam, fit_intercept=False, tol=1e-8, max_iter=10000).fit(X, y).coef_
            hits += np.array_equal(np.sign(np.where(np.abs(b) < 1e-10, 0, b)), truth)
        prob.append(hits / R)
    ax.plot(ns, prob, "-" + mk, color=col, mfc="white", mew=1.3, ms=5, label=title)
ax.set_xscale("log")
ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r"sample size $n$ (log scale)")
ax.set_ylabel(r"$\mathrm{P}\{\mathrm{sign}(\hat\beta) = \mathrm{sign}(\beta^*)\}$")
ax.legend(loc="center right", fontsize=9)
save(fig, __file__, "sign-recovery")
