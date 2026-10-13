"""Gumbel-max sampling of a categorical distribution and the Gumbel-softmax relaxation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(7)

# ---- Figure 1: argmax frequencies and the law of the maximum
a = np.array([1.2, 0.3, -0.4, 0.9, -1.5, 0.0])               # unnormalized log-probabilities
pi = np.exp(a) / np.exp(a).sum()
logS = np.log(np.exp(a).sum())
N = 200_000
Y = a + rng.gumbel(size=(N, a.size))
I, M = Y.argmax(1), Y.max(1)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
k = np.arange(1, a.size + 1)
freq = np.bincount(I, minlength=a.size) / N
ax.bar(k, pi, width=0.6, color=PALETTE["light"], edgecolor=PALETTE["grey"], label=r"$\pi_k = e^{a_k} / \sum_j e^{a_j}$")
ax.plot(k, freq, "o", color=PALETTE["blue"], ms=6, label=r"frequency of $\arg\max_j (a_j + G_j) = k$")
ax.set_xticks(k); ax.set_xlabel("category $k$"); ax.set_ylabel("probability")
ax.set_ylim(0, 0.46)
ax.set_title(rf"$2\times 10^5$ draws, $K = {a.size}$", loc="left")
ax.legend(loc="upper right", fontsize=8.8)

ax = axes[1]
x = np.linspace(-2, 8, 400)
dens = np.exp(-(x - logS) - np.exp(-(x - logS)))
bins = np.linspace(-2, 8, 70)
for j, col in [(0, PALETTE["blue"]), (4, PALETTE["red"])]:
    ax.hist(M[I == j], bins=bins, density=True, histtype="step", lw=1.4, color=col,
            label=rf"$M$ given $I = {j + 1}$ ($\pi_{j + 1} = {pi[j]:.3f}$)")
ax.plot(x, dens, color=PALETTE["ink"], lw=1.1, ls="--", label=r"Gumbel$(\log \sum_j e^{a_j})$ density")
ax.set_xlim(-2, 8); ax.set_ylim(0, 0.5)
ax.set_xlabel(r"$M = \max_j (a_j + G_j)$"); ax.set_ylabel("density")
ax.set_title("The maximum is independent of the argmax", loc="left")
ax.legend(loc="upper right", fontsize=8.8)
save(fig, __file__, "argmax-and-max")

# ---- Figure 2: Gumbel-softmax samples on the simplex for several temperatures
a3 = np.log(np.array([0.5, 0.3, 0.2]))
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])   # triangle vertices
taus = [0.1, 0.5, 1.0, 5.0]
n = 1500
fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4))
for ax, tau in zip(axes, taus):
    Y = (a3 + rng.gumbel(size=(n, 3))) / tau
    P = np.exp(Y - Y.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    xy = P @ V
    win = P.argmax(1)
    ax.plot(*np.vstack([V, V[:1]]).T, color=PALETTE["grey"], lw=0.8)
    for j in range(3):
        sel = win == j
        ax.scatter(xy[sel, 0], xy[sel, 1], s=3, color=CYCLE[j], alpha=0.5, lw=0)
    for j, (dx, dy, ha) in enumerate([(0, -0.08, "center"), (0, -0.08, "center"), (0, 0.06, "center")]):
        ax.text(V[j, 0] + dx, V[j, 1] + dy, rf"$e_{j + 1}$", ha=ha, va="center", fontsize=12.5)
    ax.set_title(rf"$\tau = {tau}$", loc="center", fontsize=13)
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.16, 1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, __file__, "gumbel-softmax", fmt="png")
