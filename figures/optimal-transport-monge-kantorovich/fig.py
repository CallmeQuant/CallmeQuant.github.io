"""Optimal transport (note: Monge and Kantorovich problems). Produces two figures:
  couplings.svg - (a) optimal assignment between two 7-point clouds in the plane (squared
                  Euclidean cost, by enumerating all permutations); (b) the monotone
                  (quantile) coupling between two densities on the line
  entropic.png  - entropic optimal transport plans between discretized versions of the same
                  two densities for three values of the regularization parameter
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
from style import use_style, PALETTE, save

use_style()
CMAP = LinearSegmentedColormap.from_list("blog", ["white", PALETTE["blue"], PALETTE["ink"]])
rng = np.random.default_rng(1781)

# ---------------------------------------------------------------- (a) assignment in the plane
n = 7
X = rng.normal(size=(n, 2)) * 0.55 + np.array([-1.3, 0.0])
ang = rng.uniform(0, 2 * np.pi, n)
Y = np.c_[np.cos(ang), np.sin(ang)] * 1.1 + np.array([1.3, 0.0]) + rng.normal(size=(n, 2)) * 0.08
C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
costs = {p: C[np.arange(n), p].sum() for p in itertools.permutations(range(n))}
best = min(costs, key=costs.get)
worst = max(costs.values())
print("optimal cost", costs[best], "mean over permutations", np.mean(list(costs.values())), "worst", worst)

# ---------------------------------------------------------------- (b) one dimension
def f_mu(x):  # mixture density
    return 0.55 * norm.pdf(x, -1.6, 0.55) + 0.45 * norm.pdf(x, 1.0, 0.7)


def f_nu(x):
    return norm.pdf(x, 0.6, 1.0)


grid = np.linspace(-5, 5, 4001)
F = np.cumsum(f_mu(grid)); F /= F[-1]
G = np.cumsum(f_nu(grid)); G /= G[-1]
Finv = lambda t: np.interp(t, F, grid)
Ginv = lambda t: np.interp(t, G, grid)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
for i, j in enumerate(best):
    a0.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], color=PALETTE["grey"], lw=1.1, zorder=1)
a0.scatter(*X.T, s=36, color=PALETTE["blue"], zorder=2, label=r"$x_i$ (mass $1/7$ each)")
a0.scatter(*Y.T, s=36, color=PALETTE["red"], marker="s", zorder=2, label=r"$y_j$ (mass $1/7$ each)")
a0.set_aspect("equal"); a0.set_xlim(-2.6, 2.8); a0.set_ylim(-1.7, 2.3)
a0.set_title(r"(a) optimal assignment, $c(x,y) = \Vert x - y\Vert^2$", loc="left")
a0.legend(loc="upper left", fontsize=8.8, handletextpad=0.3)
a0.set_xticks([-2, 0, 2]); a0.set_yticks([-1, 0, 1, 2])

xs = np.linspace(-4, 4, 600)
top, s = 1.0, 0.9                          # baseline of the upper density and its vertical scale
a1.fill_between(xs, top, top + s * f_mu(xs), color=PALETTE["blue"], alpha=0.25, lw=0)
a1.plot(xs, top + s * f_mu(xs), color=PALETTE["blue"], label=r"$\mu$")
a1.fill_between(xs, 0, -s * f_nu(xs), color=PALETTE["red"], alpha=0.25, lw=0)
a1.plot(xs, -s * f_nu(xs), color=PALETTE["red"], label=r"$\nu$")
for t in (np.arange(15) + 0.5) / 15:
    a1.annotate("", xy=(Ginv(t), 0.02), xytext=(Finv(t), top - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], lw=0.7, mutation_scale=7))
a1.axhline(top, color=PALETTE["ink"], lw=0.6); a1.axhline(0, color=PALETTE["ink"], lw=0.6)
a1.set_xlim(-4, 4); a1.set_ylim(-0.5, 1.5)
a1.set_yticks([]); a1.grid(False); a1.spines["left"].set_visible(False)
a1.set_xlabel(r"$x$")
a1.set_title(r"(b) quantile coupling $F^{-1}(t) \mapsto G^{-1}(t)$", loc="left")
a1.legend(loc="upper right")
save(fig, __file__, "couplings")

# ---------------------------------------------------------------- entropic plans (Sinkhorn)
m = 120
z = np.linspace(-4, 4, m)
a = f_mu(z); a /= a.sum()
b = f_nu(z); b /= b.sum()
Cz = (z[:, None] - z[None, :]) ** 2


def lse(M, axis):
    mx = M.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(M - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_log(a, b, C, eps, iters):
    f = np.zeros(len(a)); g = np.zeros(len(b))
    for _ in range(iters):
        f = eps * np.log(a) - eps * lse((g[None, :] - C) / eps, axis=1)
        g = eps * np.log(b) - eps * lse((f[:, None] - C) / eps, axis=0)
    return np.exp((f[:, None] + g[None, :] - C) / eps)


T_map = Ginv(np.interp(z, grid, F))
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
for ax, eps in zip(axes, [1.0, 0.1, 0.01]):
    P = sinkhorn_log(a, b, Cz, eps, 3000)
    print(eps, "marginal errors", np.abs(P.sum(1) - a).max(), np.abs(P.sum(0) - b).max(),
          "cost", (P * Cz).sum())
    ax.imshow(P.T, origin="lower", extent=[-4, 4, -4, 4], cmap=CMAP, aspect="equal",
              interpolation="nearest", vmin=0, vmax=P.max())
    ax.plot(z, T_map, color=PALETTE["red"], lw=1.0, ls="--")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_title(rf"$\varepsilon = {eps:g}$", loc="left")
    ax.set_xlabel(r"$x$ (source $\mu$)")
    ax.grid(False)
axes[0].set_ylabel(r"$y$ (target $\nu$)")
save(fig, __file__, "entropic", fmt="png")
# reference: unregularized cost of the monotone coupling on the grid
print("W2^2 (quantile formula)", np.mean((Finv((np.arange(20000) + 0.5) / 20000) - Ginv((np.arange(20000) + 0.5) / 20000)) ** 2))
