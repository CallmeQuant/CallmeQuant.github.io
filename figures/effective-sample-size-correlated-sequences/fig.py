"""Effective sample size for AR(1) sequences: exact variance inflation, and estimation of the
integrated autocorrelation time (note: effective sample size)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1992)


def ar1(phi, n, m=1):
    """m stationary AR(1) paths of length n with unit marginal variance."""
    e = rng.standard_normal((m, n)) * np.sqrt(1 - phi ** 2)
    x0 = rng.standard_normal(m)
    e[:, 0] = x0
    return lfilter([1.0], [1.0, -phi], e, axis=1)


def inflation(phi, n):
    """n Var(mean) / gamma_0 for a stationary AR(1)."""
    return (1 + phi) / (1 - phi) - 2 * phi * (1 - phi ** n) / (n * (1 - phi) ** 2)


# ---------- Figure 1: paths and variance inflation ----------
phis = [-0.5, 0.5, 0.9]
cols = [PALETTE["green"], PALETTE["blue"], PALETTE["red"]]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
for j, (phi, col) in enumerate(zip(phis, cols)):
    x = ar1(phi, 200)[0]
    ax.plot(np.arange(200), x + 5 * (2 - j), color=col, lw=0.9)
    ax.text(203, 5 * (2 - j), rf"$\phi = {phi}$", color=col, va="center", fontsize=9.5)
ax.set_xlim(0, 230); ax.set_yticks([]); ax.set_xlabel(r"$t$")
ax.set_title("Stationary AR(1) paths, unit variance", loc="left")
ax.grid(False)

ax = axes[1]
ns = np.unique(np.round(np.logspace(0, 3.3, 60)).astype(int))
for phi, col in zip(phis, cols):
    ax.plot(ns, [inflation(phi, n) for n in ns], color=col, label=rf"$\phi = {phi}$")
    ax.axhline((1 + phi) / (1 - phi), xmin=0.4, color=col, lw=0.7, ls=":")   # asymptote, drawn for n >= 20
    for n in [5, 20, 100, 500]:
        means = ar1(phi, n, 4000).mean(1)
        ax.plot(n, n * means.var(), "o", color=col, ms=4, mfc="white", mew=1.0)
ax.axhline(1, color=PALETTE["ink"], lw=0.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 2000); ax.set_ylim(0.2, 40)
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$n\,\mathrm{Var}(\bar X_n)/\gamma_0 = n/n_{\mathrm{eff}}$")
ax.set_title(r"Variance inflation approaches $\tau = (1+\phi)/(1-\phi)$", loc="left")
ax.legend(loc="upper left", fontsize=9)
save(fig, __file__, "variance-inflation")


# ---------- Figure 2: estimating tau ----------
def acov(x):
    n = len(x); y = x - x.mean()
    f = np.fft.rfft(y, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


def tau_geyer(x):
    g = acov(x)
    G = g[0:-1:2] + g[1::2]                    # Gamma_m = gamma_{2m} + gamma_{2m+1}
    M = np.argmax(G <= 0) if np.any(G <= 0) else len(G)
    return (-g[0] + 2 * G[:M].sum()) / g[0]


def tau_batch(x):
    n = len(x); m = int(np.sqrt(n)); b = n // m
    bm = x[: b * m].reshape(b, m).mean(1)
    return m * bm.var(ddof=1) / x.var()


phi, n = 0.9, 5000
tau_true = (1 + phi) / (1 - phi)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
x = ar1(phi, n)[0]
g = acov(x)
K = np.arange(n)
partial = (g[0] + 2 * np.cumsum(g[1:])) / g[0]
ax.plot(K[1:], partial, color=PALETTE["blue"], lw=1.0, label=r"$1 + 2\sum_{k \leq K}\hat\rho_k$")
ax.axhline(tau_true, color=PALETTE["red"], lw=1.0, ls="--", label=rf"true $\tau = {tau_true:.0f}$")
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.set_xscale("log"); ax.set_xlim(1, n)
ax.set_xlabel(r"truncation lag $K$"); ax.set_title(r"One chain, $\phi = 0.9$, $n = 5000$", loc="left")
ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="none", framealpha=1)

ax = axes[1]
X = ar1(phi, n, 2000)
ess_g = n / np.array([tau_geyer(r) for r in X])
ess_b = n / np.array([tau_batch(r) for r in X])
bins = np.linspace(100, 600, 51)
ax.hist(ess_b, bins=bins, color=PALETTE["gold"], alpha=0.55, label="batch means")
ax.hist(ess_g, bins=bins, color=PALETTE["blue"], alpha=0.55, label="initial positive sequence")
ax.axvline(n / tau_true, color=PALETTE["red"], lw=1.2, ls="--", label=rf"$n/\tau = {n / tau_true:.0f}$")
ax.set_xlabel(r"estimated $n_{\mathrm{eff}}$"); ax.set_ylabel("chains")
ax.set_title("Estimates from 2,000 chains", loc="left")
ax.legend(loc="upper right", fontsize=9)
save(fig, __file__, "estimating-tau")
print(np.median(ess_g), np.median(ess_b), partial[-1])
