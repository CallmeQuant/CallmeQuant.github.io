"""Riemann sums (slicing the domain) versus Lebesgue sums (slicing the range), the distribution
function t -> lambda(f > t), and dyadic simple-function approximations."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()


def f(x):
    return 1.0 + 0.5 * np.sin(2 * np.pi * x) + 0.3 * x * np.cos(7 * np.pi * x)


x = np.linspace(0, 1, 20001)
fx = f(x)
I = np.trapezoid(fx, x)
dx = x[1] - x[0]
G = lambda t: (fx >= t).mean()          # lambda(f >= t) on [0,1], grid approximation

# ---------- Figure 1 ----------
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1, 1, 0.9]})
ax = axes[0]
n = 10
edges = np.linspace(0, 1, n + 1)
for a, b in zip(edges[:-1], edges[1:]):
    h = f(a)
    ax.add_patch(plt.Rectangle((a, 0), b - a, h, fc=PALETTE["blue"], alpha=0.25, ec=PALETTE["blue"], lw=0.7))
ax.plot(x, fx, color=PALETTE["ink"], lw=1.4)
R = sum(f(a) * (b - a) for a, b in zip(edges[:-1], edges[1:]))
ax.set_title(rf"Riemann: slice the domain, sum $= {R:.3f}$", loc="left", fontsize=10.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.75); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$f(x)$")

ax = axes[1]
h = 0.25
levels = np.arange(h, fx.max() + h, h)
shades = [CYCLE[k % 5] for k in range(len(levels))]
L = 0.0
for k, y in enumerate(levels):
    mask = fx >= y
    if not mask.any():
        continue
    L += h * mask.mean()
    # draw the set {f >= y} as a horizontal band between y - h and y
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    for s, e in zip(idx[::2], idx[1::2]):
        ax.add_patch(plt.Rectangle((x[s], y - h), x[e - 1] - x[s], h, fc=shades[k], alpha=0.3, ec="none"))
        ax.plot([x[s], x[e - 1]], [y, y], color=shades[k], lw=2.0, solid_capstyle="butt")
ax.plot(x, fx, color=PALETTE["ink"], lw=1.4)
ax.set_title(rf"Lebesgue: slice the range, sum $= {L:.3f}$", loc="left", fontsize=10.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.75); ax.set_xlabel(r"$x$")

ax = axes[2]
t = np.linspace(0, 1.75, 600)
Gt = np.array([G(s) for s in t])
ax.fill_betweenx(t, 0, Gt, color=PALETTE["grey"], alpha=0.3, lw=0)
ax.plot(Gt, t, color=PALETTE["ink"], lw=1.4)
for k, y in enumerate(levels):
    ax.plot([0, G(y)], [y, y], color=shades[k], lw=2.0)
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.75)
ax.set_xlabel(r"$\lambda(\{f \geq t\})$"); ax.set_ylabel(r"$t$")
ax.set_title(rf"Area $= \int f\,d\lambda = {I:.3f}$", loc="left", fontsize=10.5)
save(fig, __file__, "riemann-vs-lebesgue")

# ---------- Figure 2: dyadic simple functions phi_n ----------
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.8), sharey=True)
for j, nn in enumerate([1, 2, 3, 4]):
    ax = axes[j]
    phi = np.minimum(np.floor(fx * 2 ** nn) / 2 ** nn, nn)
    sl = slice(None, None, 5)                  # thin the grid for plotting only
    ax.fill_between(x[sl], 0, phi[sl], color=PALETTE["blue"], alpha=0.25, lw=0)
    ax.plot(x[sl], phi[sl], color=PALETTE["blue"], lw=1.0)
    ax.plot(x[sl], fx[sl], color=PALETTE["ink"], lw=1.2)
    for k in range(1, int(nn * 2 ** nn) + 1):
        y = k / 2 ** nn
        if y <= 1.75:
            ax.axhline(y, color=PALETTE["grey"], lw=0.4, alpha=0.6)
    ax.set_title(rf"$n = {nn}$: $\int\varphi_{nn} = {phi.mean():.3f}$", loc="left", fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.75); ax.set_xlabel(r"$x$")
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.grid(axis="y", visible=False)
axes[0].set_ylabel(r"$\varphi_n(x)$")
save(fig, __file__, "dyadic-approximations")
print(I, R, L)
