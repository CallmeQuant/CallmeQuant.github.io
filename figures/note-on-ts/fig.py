"""Sample paths and sample autocorrelation functions for Gaussian white noise, an AR(1) process
with phi = 0.7 and an MA(1) process with theta = 0.8 (note: note-on-ts)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(3)
n, H, burn = 200, 20, 500
phi, theta = 0.7, 0.8

z = rng.normal(size=n + burn)
ar = np.zeros(n + burn)
for t in range(1, n + burn):
    ar[t] = phi * ar[t - 1] + z[t]
series = {
    "white noise": (z[burn:], np.zeros(H)),
    r"AR(1), $\phi = 0.7$": (ar[burn:], phi ** np.arange(1, H + 1)),
    r"MA(1), $\theta = 0.8$": (z[burn:] + theta * z[burn - 1:-1],
                               np.r_[theta / (1 + theta**2), np.zeros(H - 1)]),
}

def sample_acf(x, H):
    xc = x - x.mean()
    g0 = xc @ xc / len(x)
    return np.array([xc[h:] @ xc[:-h] / len(x) / g0 for h in range(1, H + 1)])

fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.6), sharey="row",
                         gridspec_kw={"height_ratios": [1, 1.15]})
lags = np.arange(1, H + 1)
band = 1.96 / np.sqrt(n)
for j, (name, (x, rho)) in enumerate(series.items()):
    ax = axes[0, j]
    ax.plot(np.arange(1, n + 1), x, color=PALETTE["blue"], lw=0.8)
    ax.set_title(name, loc="left")
    ax.set_xlabel(r"$t$")
    ax = axes[1, j]
    r = sample_acf(x, H)
    ax.axhspan(-band, band, color=PALETTE["light"], lw=0, zorder=0)
    ax.axhline(0, color=PALETTE["ink"], lw=0.6)
    ax.vlines(lags, 0, r, color=PALETTE["blue"], lw=1.6, label=r"sample $\hat\rho(h)$")
    ax.scatter(lags, rho, s=16, facecolor="white", edgecolor=PALETTE["red"], lw=1.2, zorder=3,
               label=r"true $\rho(h)$")
    ax.set_xlabel(r"lag $h$")
    ax.set_ylim(-0.35, 0.85)
    ax.set_xticks([1, 5, 10, 15, 20])
axes[0, 0].set_ylabel(r"$X_t$")
axes[1, 0].set_ylabel("autocorrelation")
axes[1, 2].legend(loc="upper right")
save(fig, __file__, "sample-acf")
