"""Maximum likelihood under misspecification (note: fitting the wrong model).
Figure 1: a normal linear model fitted to a nonlinear regression; the KL projection depends on the design.
Figure 2: exponential model fitted to log-normal data; coverage of model-based and sandwich intervals."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1982)

# ---- Figure 1 ----
m = lambda z: 2 * np.sin(z)
designs = [("Design A: $Z \\sim N(0,1)$", 0.0, 1.0), ("Design B: $Z \\sim N(1.5,\\,0.5^2)$", 1.5, 0.5)]
n = 300
zz = np.linspace(-3.2, 3.2, 400)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
for ax, (title, mu, sd) in zip(axes, designs):
    z = rng.normal(mu, sd, n)
    y = m(z) + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), z])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    # population projection: slope = Cov(Z, m(Z))/Var(Z), by Stein's lemma = E m'(Z) = 2 E cos Z
    slope = 2 * np.cos(mu) * np.exp(-sd**2 / 2)
    inter = 2 * np.sin(mu) * np.exp(-sd**2 / 2) - slope * mu
    ax.scatter(z, y, s=7, color=PALETTE["grey"], alpha=0.55, lw=0)
    ax.plot(zz, m(zz), color=PALETTE["ink"], lw=1.5, label=r"$\mathrm{E}(Y \mid Z = z) = 2\sin z$")
    ax.plot(zz, inter + slope * zz, color=PALETTE["blue"], lw=1.8, label=r"projection $\beta_0^* + \beta_1^* z$")
    ax.plot(zz, b[0] + b[1] * zz, color=PALETTE["red"], lw=1.2, ls="--", label=r"fit $\hat\beta_0 + \hat\beta_1 z$, $n = 300$")
    dens = stats.norm.pdf(zz, mu, sd)
    ax.fill_between(zz, -4.6, -4.6 + 1.3 * dens / dens.max(), color=PALETTE["gold"], alpha=0.35, lw=0)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-4.6, 4.6)
    ax.set_xlabel(r"$z$")
    ax.set_title(title + rf", $\beta_1^* = {slope:.2f}$", loc="left", fontsize=10.5)
axes[0].set_ylabel(r"$y$")
axes[0].legend(loc="upper left", fontsize=8.4)
save(fig, __file__, "design-dependence")

# ---- Figure 2 ----
ss = np.linspace(0.2, 1.6, 15)
z975 = stats.norm.ppf(0.975)
sfine = np.linspace(0.15, 1.65, 300)
cv = np.sqrt(np.expm1(sfine**2))
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.plot(sfine, 2 * stats.norm.cdf(z975 / cv) - 1, color=PALETTE["red"], lw=1.6,
        label=r"model-based, $n \to \infty$: $2\Phi(1.96/\mathrm{CV}) - 1$")
ax.axhline(0.95, color=PALETTE["blue"], lw=1.6, label=r"sandwich, $n \to \infty$")
R = 20000
for nn, mk in [(100, "o"), (1000, "s")]:
    cov_m, cov_s = [], []
    for s in ss:
        lam_star = np.exp(-s**2 / 2)          # 1 / E X for X ~ LogNormal(0, s^2)
        x = rng.lognormal(0.0, s, (R, nn))
        xbar = x.mean(axis=1)
        lam = 1 / xbar
        se_m = lam / np.sqrt(nn)
        se_s = np.sqrt(x.var(axis=1, ddof=1) / nn) / xbar**2
        cov_m.append(np.mean(np.abs(lam - lam_star) <= z975 * se_m))
        cov_s.append(np.mean(np.abs(lam - lam_star) <= z975 * se_s))
    ax.plot(ss, cov_m, mk, color=PALETTE["red"], mfc="white", mew=1.2, ms=4.5, label=rf"model-based, $n = {nn}$")
    ax.plot(ss, cov_s, mk, color=PALETTE["blue"], mfc="white", mew=1.2, ms=4.5, label=rf"sandwich, $n = {nn}$")
ax.axvline(np.sqrt(np.log(2)), color=PALETTE["grey"], lw=0.8, ls=":")
ax.text(np.sqrt(np.log(2)) + 0.02, 0.76, r"$\mathrm{CV} = 1$", color=PALETTE["grey"], fontsize=9.5)
ax.set_xlim(0.15, 1.65); ax.set_ylim(0.38, 1.02)
ax.set_xlabel(r"log-normal shape $s$")
ax.set_ylabel("coverage of nominal 95% interval")
ax.legend(loc="lower left", fontsize=8.4, ncol=1)
save(fig, __file__, "coverage")
