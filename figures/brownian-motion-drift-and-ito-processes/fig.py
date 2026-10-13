"""Wiener process, Brownian motion with drift, and an Ornstein-Uhlenbeck (Ito) process;
variation of Brownian paths on finer grids."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1923)
T, N, paths = 4.0, 2000, 5
dt = T / N
t = np.linspace(0, T, N + 1)
dW = rng.normal(scale=np.sqrt(dt), size=(paths, N))
W = np.hstack([np.zeros((paths, 1)), np.cumsum(dW, axis=1)])

a, b = 0.75, 0.6                              # drift and scale of the generalized Wiener process
X = a * t + b * W
kappa, sigma, x0 = 2.0, 0.8, 1.5              # OU parameters
Y = np.empty_like(W); Y[:, 0] = x0
for k in range(N):                            # Euler-Maruyama
    Y[:, k + 1] = Y[:, k] - kappa * Y[:, k] * dt + sigma * dW[:, k]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True)
ax = axes[0]
ax.fill_between(t, -2 * np.sqrt(t), 2 * np.sqrt(t), color=PALETTE["light"], lw=0)
ax.plot(t, 0 * t, color=PALETTE["ink"], lw=1.0, ls="--")
for i in range(paths):
    ax.plot(t, W[i], color=CYCLE[i % 5], lw=0.8)
ax.set_title(r"$W_t$: Wiener process", loc="left")
lim = max(4.4, 1.05 * np.abs(W).max()); ax.set_ylim(-lim, lim)

ax = axes[1]
ax.fill_between(t, a * t - 2 * b * np.sqrt(t), a * t + 2 * b * np.sqrt(t), color=PALETTE["light"], lw=0)
ax.plot(t, a * t, color=PALETTE["ink"], lw=1.0, ls="--")
for i in range(paths):
    ax.plot(t, X[i], color=CYCLE[i % 5], lw=0.8)
ax.set_title(rf"$X_t = {a}\,t + {b}\,W_t$", loc="left")
ax.set_ylim(min(-1.8, 1.05 * X.min()), max(5.8, 1.05 * X.max()))

ax = axes[2]
m = x0 * np.exp(-kappa * t)
s = sigma * np.sqrt((1 - np.exp(-2 * kappa * t)) / (2 * kappa))
ax.fill_between(t, m - 2 * s, m + 2 * s, color=PALETTE["light"], lw=0)
ax.plot(t, m, color=PALETTE["ink"], lw=1.0, ls="--")
for i in range(paths):
    ax.plot(t, Y[i], color=CYCLE[i % 5], lw=0.8)
ax.set_title(rf"$dY_t = -{kappa:g}\,Y_t\,dt + {sigma}\,dW_t$", loc="left")
ax.set_ylim(min(-1.6, 1.05 * Y.min()), max(2.0, 1.05 * Y.max()))
for ax in axes:
    ax.set_xlim(0, T); ax.set_xlabel(r"$t$")
save(fig, __file__, "three-processes")

# ---------- variation sums on dyadic grids ----------
K = 2 ** 20
inc = rng.normal(scale=np.sqrt(1.0 / K), size=(3, K))
levels = np.arange(2, 21)
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
for i in range(3):
    path = np.concatenate([[0.0], np.cumsum(inc[i])])
    qv, tv = [], []
    for l in levels:
        d = np.diff(path[:: 2 ** (20 - l)])
        qv.append((d ** 2).sum()); tv.append(np.abs(d).sum())
    axes[0].plot(2.0 ** levels, qv, "o-", ms=3, lw=1.0, color=CYCLE[i])
    axes[1].plot(2.0 ** levels, tv, "o-", ms=3, lw=1.0, color=CYCLE[i])
n = 2.0 ** levels
axes[0].axhline(1.0, color=PALETTE["ink"], lw=0.8, ls="--")
axes[1].plot(n, np.sqrt(2 * n / np.pi), color=PALETTE["ink"], lw=0.8, ls="--", label=r"$\sqrt{2n/\pi}$")
axes[0].set_title("Sum of squared increments on [0, 1]", loc="left")
axes[1].set_title("Sum of absolute increments on [0, 1]", loc="left")
axes[0].set_ylim(0, 2.6)
axes[1].set_yscale("log")
axes[1].legend(loc="upper left")
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xlabel(r"number of intervals $n$")
save(fig, __file__, "variation")
