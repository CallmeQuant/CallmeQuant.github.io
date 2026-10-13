"""Inducing-point approximations to Gaussian process regression: SoR, DTC and FITC predictions against
the exact posterior, and their accuracy as the number of inducing points grows
(note: inducing points for Gaussian processes)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2005)
ELL, SF, SN = 0.8, 1.0, 0.3


def kern(a, b):
    return SF ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ELL ** 2)


def f_true(x):
    return np.sin(x) + 0.5 * np.sin(3 * x)


n = 300
X = np.concatenate([rng.uniform(0, 4, n // 2), rng.uniform(6, 10, n // 2)])
y = f_true(X) + SN * rng.standard_normal(n)
xs = np.linspace(-2.5, 12.5, 500)


def exact(xq):
    c = cho_factor(kern(X, X) + SN ** 2 * np.eye(n), lower=True)
    Ks = kern(xq, X)
    return Ks @ cho_solve(c, y), SF ** 2 - np.sum(Ks * cho_solve(c, Ks.T).T, axis=1)


def sparse(xq, Z, method):
    """Predictive mean and variance of f(xq) under SoR, DTC or FITC with inducing inputs Z (O(n m^2))."""
    m = Z.size
    Kuu = kern(Z, Z) + 1e-8 * np.eye(m)
    Kuf = kern(Z, X)
    Ksu = kern(xq, Z)
    cu = cho_factor(Kuu, lower=True)
    if method == "FITC":
        qdiag = np.sum(Kuf * cho_solve(cu, Kuf), axis=0)
        lam = SF ** 2 - qdiag + SN ** 2                     # heteroscedastic training noise
    else:
        lam = np.full(n, SN ** 2)
    A = Kuu + (Kuf / lam) @ Kuf.T                           # Sigma^{-1}
    ca = cho_factor(A, lower=True)
    mean = Ksu @ cho_solve(ca, Kuf @ (y / lam))
    var_w = np.sum(Ksu * cho_solve(ca, Ksu.T).T, axis=1)     # k_*u Sigma k_u*
    if method == "SoR":
        var = var_w
    else:
        qss = np.sum(Ksu * cho_solve(cu, Ksu.T).T, axis=1)
        var = SF ** 2 - qss + var_w
    return mean, var


Z = np.linspace(0.3, 9.7, 12)
fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.9), sharex=True, sharey=True)
panels = [("exact posterior", None), ("SoR", "SoR"), ("DTC", "DTC"), ("FITC", "FITC")]
for ax, (title, meth) in zip(axes.ravel(), panels):
    mu, var = exact(xs) if meth is None else sparse(xs, Z, meth)
    sd = np.sqrt(np.clip(var, 0, None))
    ax.plot(X, y, ".", color=PALETTE["grey"], ms=2.2, alpha=0.6)
    ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color=PALETTE["blue"], alpha=0.18, lw=0)
    ax.plot(xs, mu, color=PALETTE["blue"], lw=1.6)
    ax.plot(xs, f_true(xs), color=PALETTE["ink"], ls="--", lw=0.9)
    if meth is not None:
        ax.plot(Z, np.full(Z.size, -2.55), "^", color=PALETTE["red"], ms=5, clip_on=False)
    ax.set_title(title, loc="left")
    ax.set_xlim(-2.5, 12.5)
    ax.set_ylim(-2.7, 2.7)
for ax in axes[1]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$f(x)$")
save(fig, __file__, "sparse-predictions")

# ------------- accuracy against m -------------------------------------------------------------
xt = np.linspace(-2.5, 12.5, 400)
yt = f_true(xt) + SN * rng.standard_normal(xt.size)
mu_ex, var_ex = exact(xt)
lpd_ex = np.mean(norm.logpdf(yt, mu_ex, np.sqrt(var_ex + SN ** 2)))
ms = np.arange(2, 41)
res = {k: [] for k in ("SoR", "DTC", "FITC")}
for m in ms:
    Zm = np.linspace(0.3, 9.7, m)
    for k in res:
        mu, var = sparse(xt, Zm, k)
        res[k].append((np.sqrt(np.mean((np.sqrt(np.clip(var, 0, None)) - np.sqrt(var_ex)) ** 2)),
                       np.mean(norm.logpdf(yt, mu, np.sqrt(np.clip(var, 0, None) + SN ** 2)))))
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
for c, k in zip([PALETTE["red"], PALETTE["blue"], PALETTE["green"]], res):
    r = np.array(res[k])
    ls = "--" if k == "DTC" else "-"
    axes[0].semilogy(ms, r[:, 0], color=c, ls=ls, label=k)
    axes[1].plot(ms, r[:, 1], color=c, ls=ls, label=k)
axes[1].axhline(lpd_ex, color=PALETTE["ink"], lw=1.0, ls=":", label="exact GP")
axes[0].set_xlabel(r"number of inducing points $m$")
axes[0].set_ylabel("RMS difference of sd")
axes[0].set_title("Posterior sd of $f$ vs. exact", loc="left")
axes[0].legend(loc="center right")
axes[1].set_xlabel(r"number of inducing points $m$")
axes[1].set_ylabel("mean log predictive density")
axes[1].set_title("Held-out predictive density", loc="left")
axes[1].set_ylim(-4, 0)
axes[1].legend(loc="lower right")
for ax in axes:
    ax.set_xlim(2, 40)
save(fig, __file__, "accuracy-vs-m")
print("exact lpd", lpd_ex, {k: np.round(np.array(v)[[4, 10, 20, 38], 1], 3) for k, v in res.items()})
