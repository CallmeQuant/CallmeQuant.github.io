"""Variance of score-function and pathwise gradient estimators for d/dtheta E[X^2], X ~ N(theta, sigma^2)
(note: the score-function gradient estimator)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
sigma = 1.0
rng = np.random.default_rng(12)


def estimators(theta, z):
    """Single-sample estimators of the gradient 2*theta, from standard normal draws z."""
    x = theta + sigma * z
    score = z / sigma                                       # d/dtheta log p_theta(x)
    return {
        "sf": x**2 * score,
        "mean": (x**2 - (theta**2 + sigma**2)) * score,
        "opt": (x**2 - (theta**2 + 3 * sigma**2)) * score,
        "path": 2 * x,
    }


exact = {
    "sf": lambda t: t**4 / sigma**2 + 14 * t**2 + 15 * sigma**2,
    "mean": lambda t: 8 * t**2 + 10 * sigma**2,
    "opt": lambda t: 8 * t**2 + 6 * sigma**2,
    "path": lambda t: 4 * sigma**2 + 0 * t,
}
names = {
    "sf": "score function, no baseline",
    "mean": r"baseline $b = \mathbb{E} f(X)$",
    "opt": r"optimal baseline $b^*$",
    "path": "pathwise (reparameterization)",
}
cols = {"sf": PALETTE["red"], "mean": PALETTE["gold"], "opt": PALETTE["green"], "path": PALETTE["blue"]}

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
t = np.linspace(0, 4, 200)
te = np.arange(0, 4.01, 0.5)
z = rng.standard_normal(200_000)
for k in exact:
    ax.plot(t, exact[k](t), color=cols[k], label=names[k])
    ax.plot(te, [estimators(th, z)[k].var() for th in te], "o", color=cols[k], ms=3.5)
ax.set_yscale("log")
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("variance of one-sample estimator")
ax.set_title(r"$f(x) = x^2$, $X \sim N(\theta, 1)$", loc="left")
ax.set_xlim(0, 4); ax.set_ylim(2, 2e3)
ax.legend(loc="upper left", fontsize=8.8)

ax = axes[1]
theta, n = 2.0, 5000
z = rng.standard_normal(n)
est = estimators(theta, z)
m = np.arange(1, n + 1)
for k in ["sf", "opt", "path"]:
    half = 2 * np.sqrt(exact[k](theta) / m)
    ax.fill_between(m, 2 * theta - half, 2 * theta + half, color=cols[k], alpha=0.12, lw=0)
    ax.plot(m, np.cumsum(est[k]) / m, color=cols[k], lw=1.1)
ax.axhline(2 * theta, color=PALETTE["ink"], lw=0.7, ls="--")
ax.set_xscale("log"); ax.set_xlim(5, n); ax.set_ylim(-4, 12)
ax.set_xlabel("number of samples $n$"); ax.set_ylabel(r"running average of estimates")
ax.set_title(r"Estimating $\nabla L(2) = 4$, bands $\pm 2\,\mathrm{sd}/\sqrt{n}$", loc="left")
save(fig, __file__, "gradient-variance")
