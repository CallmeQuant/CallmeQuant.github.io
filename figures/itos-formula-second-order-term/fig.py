"""Riemann sums for int W dW with different evaluation points; geometric Brownian motion
(note: Ito's formula)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1944)

# ---------- Figure 1: left, midpoint and right sums for int_0^1 W dW ----------
L = 18
K = 2 ** L
W = np.concatenate([[0.0], np.cumsum(rng.normal(scale=np.sqrt(1 / K), size=K))])
levels = np.arange(1, L)                      # n = 2^l intervals, midpoints on the finer grid
sums = {"left": [], "mid": [], "right": []}
for l in levels:
    step = 2 ** (L - l)
    Wg = W[::step]; dW = np.diff(Wg); Wm = W[step // 2::step][: len(dW)]
    sums["left"].append((Wg[:-1] * dW).sum())
    sums["mid"].append((Wm * dW).sum())
    sums["right"].append((Wg[1:] * dW).sum())
w1 = W[-1]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
n = 2.0 ** levels
spec = [("right", r"right endpoint $\to \frac{1}{2}(W_1^2 + 1)$", PALETTE["red"], 0.5 * (w1 ** 2 + 1)),
        ("mid", r"midpoint $\to \frac{1}{2} W_1^2$", PALETTE["green"], 0.5 * w1 ** 2),
        ("left", r"left endpoint $\to \frac{1}{2}(W_1^2 - 1)$", PALETTE["blue"], 0.5 * (w1 ** 2 - 1))]
for key, lab, col, lim in spec:
    ax.axhline(lim, color=col, lw=0.8, ls="--")
    ax.plot(n, sums[key], "o-", color=col, ms=3, lw=1.0, label=lab)
ax.set_xscale("log", base=2)
ax.set_xlabel(r"number of intervals $n$")
ax.set_title(r"Sums $\sum_i W_{s_i}\,(W_{t_i} - W_{t_{i-1}})$, one path", loc="left")
ax.legend(loc="upper center", fontsize=8.8)
ax.set_ylim(min(spec[2][3], min(sums["left"])) - 0.35, max(spec[0][3], max(sums["right"])) + 0.95)

ax = axes[1]
M, nint = 3000, 500
dWs = rng.normal(scale=np.sqrt(1 / nint), size=(M, nint))
Ws = np.cumsum(dWs, axis=1)
left = (np.hstack([np.zeros((M, 1)), Ws[:, :-1]]) * dWs).sum(1)
ax.scatter(Ws[:, -1], left, s=3, color=PALETTE["blue"], alpha=0.35, lw=0, rasterized=True,
           label=r"left sums, $n = 500$")
x = np.linspace(-3.5, 3.5, 300)
ax.plot(x, 0.5 * x ** 2, color=PALETTE["grey"], lw=1.2, label=r"$\frac{1}{2} w^2$ (ordinary calculus)")
ax.plot(x, 0.5 * (x ** 2 - 1), color=PALETTE["ink"], lw=1.2, ls="--", label=r"$\frac{1}{2} (w^2 - 1)$")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1.0, 5.5)
ax.set_xlabel(r"$W_1$"); ax.set_title(r"Left sums across 3,000 paths", loc="left")
ax.legend(loc="upper center", fontsize=8.8, markerscale=3)
save(fig, __file__, "riemann-sums")

# ---------- Figure 2: geometric Brownian motion, mean versus typical path ----------
mu, sigma, T, N, P = 0.1, 0.6, 10.0, 1000, 2000
t = np.linspace(0, T, N + 1)
Wp = np.hstack([np.zeros((P, 1)), np.cumsum(rng.normal(scale=np.sqrt(T / N), size=(P, N)), axis=1)])
S = np.exp((mu - sigma ** 2 / 2) * t + sigma * Wp)
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for i in range(25):
    ax.plot(t, S[i], color=PALETTE["grey"], lw=0.5, alpha=0.7)
q = np.quantile(S, [0.1, 0.9], axis=0)
ax.plot(t, q[0], color=PALETTE["violet"], lw=1.0, ls=":")
ax.plot(t, q[1], color=PALETTE["violet"], lw=1.0, ls=":", label="10% and 90% quantiles")
ax.plot(t, np.exp(mu * t), color=PALETTE["red"], lw=2, label=r"mean $e^{\mu t}$")
ax.plot(t, np.exp((mu - sigma ** 2 / 2) * t), color=PALETTE["blue"], lw=2,
        label=r"median $e^{(\mu - \sigma^2/2) t}$")
ax.plot(t, S.mean(0), color=PALETTE["red"], lw=1.0, ls="--", label="sample mean of 2,000 paths")
ax.set_yscale("log"); ax.set_xlim(0, T); ax.set_ylim(1e-3, 1e2)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S_t$ (log scale)")
ax.set_title(rf"$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$, $\mu = {mu}$, $\sigma = {sigma}$, $S_0 = 1$", loc="left")
ax.legend(loc="lower left", fontsize=8.8, ncol=2)
save(fig, __file__, "gbm")
