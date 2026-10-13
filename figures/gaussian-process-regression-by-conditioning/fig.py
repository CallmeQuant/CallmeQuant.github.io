"""Gaussian process regression: prior draws for three kernels, the posterior given noisy data, and the
log marginal likelihood as a function of the length-scale (note: Gaussian process regression by conditioning)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1996)


def k_se(a, b, ell, sf=1.0):
    return sf ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def k_exp(a, b, ell, sf=1.0):
    return sf ** 2 * np.exp(-np.abs(a[:, None] - b[None, :]) / ell)


# ---------------- Figure 1: prior draws ----------------------------------------------------------
xs = np.linspace(-3, 3, 400)
kernels = [(r"squared exponential, $\ell = 1$", lambda a, b: k_se(a, b, 1.0)),
           (r"squared exponential, $\ell = 0.25$", lambda a, b: k_se(a, b, 0.25)),
           (r"exponential, $\ell = 1$", lambda a, b: k_exp(a, b, 1.0))]
fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), sharey=True)
for ax, (title, kf) in zip(axes, kernels):
    K = kf(xs, xs) + 1e-9 * np.eye(xs.size)
    L = np.linalg.cholesky(K)
    ax.fill_between(xs, -2, 2, color=PALETTE["light"], lw=0)
    for j in range(4):
        ax.plot(xs, L @ rng.standard_normal(xs.size), color=CYCLE[j], lw=1.1)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel(r"$x$")
axes[0].set_ylabel(r"$f(x)$")
save(fig, __file__, "prior-draws")

# ---------------- data and marginal likelihood -----------------------------------------------
def f_true(x):
    return np.sin(1.5 * x) + 0.3 * x


n, sigma = 15, 0.25
X = np.sort(rng.uniform(-3, 3, n))
y = f_true(X) + sigma * rng.standard_normal(n)


def nlml_parts(ell, sf, sn):
    Ky = k_se(X, X, ell, sf) + sn ** 2 * np.eye(n)
    c = cho_factor(Ky, lower=True)
    alpha = cho_solve(c, y)
    fit = 0.5 * y @ alpha
    comp = np.sum(np.log(np.diag(c[0])))
    return fit, comp, 0.5 * n * np.log(2 * np.pi)


def nlml(p):
    return sum(nlml_parts(*np.exp(p)))


best = min((minimize(nlml, p0, method="L-BFGS-B") for p0 in
            [np.log([0.5, 1.0, 0.3]), np.log([2.0, 1.0, 0.1]), np.log([0.2, 0.5, 0.5])]), key=lambda r: r.fun)
ell_hat, sf_hat, sn_hat = np.exp(best.x)
print(f"type-II ML: ell={ell_hat:.3f}, sf={sf_hat:.3f}, sigma={sn_hat:.3f}, log p(y)={-best.fun:.3f}")


def posterior(xq, ell, sf, sn):
    Ky = k_se(X, X, ell, sf) + sn ** 2 * np.eye(n)
    c = cho_factor(Ky, lower=True)
    Ks = k_se(xq, X, ell, sf)
    mean = Ks @ cho_solve(c, y)
    cov = k_se(xq, xq, ell, sf) - Ks @ cho_solve(c, Ks.T)
    return mean, cov


# ---------------- Figure 2: posterior -----------------------------------------------------------
mean, cov = posterior(xs, ell_hat, sf_hat, sn_hat)
sd = np.sqrt(np.clip(np.diag(cov), 0, None))
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd, color=PALETTE["blue"], alpha=0.15, lw=0,
                label=r"95% pointwise band for $f$")
Lp = np.linalg.cholesky(cov + 1e-8 * np.eye(xs.size))
for j in range(3):
    ax.plot(xs, mean + Lp @ rng.standard_normal(xs.size), color=PALETTE["blue"], lw=0.7, alpha=0.6,
            label="posterior draws" if j == 0 else None)
ax.plot(xs, mean, color=PALETTE["blue"], lw=1.9, label="posterior mean")
ax.plot(xs, f_true(xs), color=PALETTE["ink"], ls="--", lw=1.2, label=r"true $f$")
ax.plot(X, y, "o", color=PALETTE["red"], ms=4.5, label="observations")
ax.set_xlim(-3, 3)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.legend(loc="upper left", ncol=2, fontsize=8.8)
save(fig, __file__, "posterior")

# ---------------- Figure 3: marginal likelihood in the length-scale -----------------------------
ells = np.geomspace(0.05, 10, 300)
parts = np.array([nlml_parts(l, sf_hat, sn_hat) for l in ells])
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
ax.plot(ells, -parts.sum(1), color=PALETTE["ink"], label=r"$\log p(y \mid \ell)$")
ax.plot(ells, -parts[:, 0], color=PALETTE["red"], ls="--", label=r"$-\frac{1}{2} y^\top K_y^{-1} y$ (fit)")
ax.plot(ells, -parts[:, 1], color=PALETTE["blue"], ls="--", label=r"$-\frac{1}{2}\log\det K_y$ (complexity)")
ax.axvline(ell_hat, color=PALETTE["grey"], lw=0.9, ls=":")
ax.set_xscale("log")
ax.set_xlim(0.05, 10)
ax.set_ylim(-40, 25)
ax.set_xlabel(r"length-scale $\ell$")
ax.set_ylabel("nats")
ax.set_title(r"Marginal likelihood ($\sigma_f$, $\sigma$ fixed)")
ax.legend(loc="lower left", fontsize=8.6)
ax = axes[1]
for k, (l, lab) in enumerate([(0.08, r"$\ell = 0.08$"), (ell_hat, rf"$\hat\ell = {ell_hat:.2f}$"), (6.0, r"$\ell = 6$")]):
    m_, _ = posterior(xs, l, sf_hat, sn_hat)
    ax.plot(xs, m_, color=CYCLE[[3, 0, 2][k]], lw=1.5, label=lab)
ax.plot(X, y, "o", color=PALETTE["red"], ms=4)
ax.set_xlim(-3, 3)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"$x$")
ax.set_title("Posterior means")
ax.legend(loc="upper left", fontsize=8.6)
save(fig, __file__, "marginal-likelihood")
