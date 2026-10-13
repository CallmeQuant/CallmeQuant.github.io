"""Kernel ridge regression: the fitted function as a sum of kernel sections at the data, and choice of the
penalty by the closed-form leave-one-out formula (note: the representer theorem and kernel ridge regression)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1971)
WIDTH = 0.08


def kern(a, b):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / WIDTH ** 2)


def f_true(x):
    return x * np.sin(3 * np.pi * x) + 0.3


n = 40
X = np.sort(rng.uniform(0, 1, n))
y = f_true(X) + 0.2 * rng.standard_normal(n)
K = kern(X, X)
xs = np.linspace(0, 1, 500)
Ks = kern(xs, X)


def fit(lam):
    alpha = np.linalg.solve(K + n * lam * np.eye(n), y)
    return alpha


def loo(lam):
    H = K @ np.linalg.inv(K + n * lam * np.eye(n))
    r = (y - H @ y) / (1 - np.diag(H))
    return np.mean(r ** 2), np.trace(H)


def loo_brute(lam):
    errs = []
    for i in range(n):
        keep = np.arange(n) != i
        a = np.linalg.solve(K[np.ix_(keep, keep)] + n * lam * np.eye(n - 1), y[keep])
        errs.append(y[i] - K[i, keep] @ a)
    return np.mean(np.square(errs))


lams = np.geomspace(1e-6, 1e0, 120)
cv = np.array([loo(l) for l in lams])
lam_hat = lams[np.argmin(cv[:, 0])]
print(f"LOO-optimal lambda = {lam_hat:.2e}, df = {loo(lam_hat)[1]:.2f}")

# ---------------- Figure 1: the fit as a sum of kernel sections -------------------------------
alpha = fit(lam_hat)
fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
ax = axes[0]
ax.plot(xs, f_true(xs), color=PALETTE["ink"], ls="--", lw=1.0, label="true regression function")
ax.plot(xs, Ks @ alpha, color=PALETTE["blue"], lw=1.9, label=r"$\hat f = \sum_i \hat\alpha_i k(\cdot, x_i)$")
ax.plot(X, y, "o", color=PALETTE["red"], ms=3.8, label="data")
ax.set_ylim(-1.3, 1.6)
ax.set_ylabel(r"$y$")
ax.legend(loc="lower left", fontsize=8.8, ncol=3)
ax = axes[1]
for i in range(n):
    ax.plot(xs, alpha[i] * Ks[:, i], color=CYCLE[i % 4], lw=0.9, alpha=0.85)
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.vlines(X, 0, alpha * 1.0, color=PALETTE["grey"], lw=0.8)
ax.set_xlim(0, 1)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\hat\alpha_i\, k(x, x_i)$")
save(fig, __file__, "kernel-sections")

# ---------------- Figure 2: penalty selection ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axes[0]
for k, (lam, lab) in enumerate([(1e-6, r"$\lambda = 10^{-6}$"), (lam_hat, rf"$\hat\lambda = {lam_hat * 100:.2f} \times 10^{{-2}}$"),
                                (0.3, r"$\lambda = 0.3$")]):
    ax.plot(xs, Ks @ fit(lam), color=CYCLE[[3, 0, 2][k]], lw=1.5, label=lab)
ax.plot(X, y, "o", color=PALETTE["red"], ms=3.5)
ax.set_xlim(0, 1)
ax.set_ylim(-1.3, 1.6)
ax.set_xlabel(r"$x$")
ax.set_title("Fits for three penalties", loc="left")
ax.legend(loc="lower left", fontsize=8.6)
ax = axes[1]
ax.plot(lams, cv[:, 0], color=PALETTE["blue"], lw=1.8, label="closed-form LOO")
check = lams[::12]
ax.plot(check, [loo_brute(l) for l in check], "o", mfc="white", color=PALETTE["red"], ms=5, label="refitting $n$ times")
ax.axvline(lam_hat, color=PALETTE["grey"], ls=":", lw=0.9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"penalty $\lambda$")
ax.set_ylabel("leave-one-out mean squared error")
ax.set_title("Leave-one-out error", loc="left")
ax.legend(loc="upper left", fontsize=8.6)
secx = ax.secondary_xaxis("top", functions=(lambda l: l, lambda l: l))
ticks = [1e-5, 1e-3, 1e-1]
secx.set_xticks(ticks)
secx.xaxis.set_minor_locator(plt.NullLocator())
secx.set_xticklabels([f"{loo(t)[1]:.0f}" for t in ticks])
secx.set_xlabel(r"degrees of freedom $\mathrm{tr}\,H_\lambda$", fontsize=9.5)
save(fig, __file__, "penalty-selection")
