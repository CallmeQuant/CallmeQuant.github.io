"""Chi lengths of projections of a standard Gaussian vector; chi densities; the t statistic as a
cotangent (note: chi lengths and Gaussian right triangles)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi, t as student
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1908)
n, k, M = 10, 3, 1500
Z = rng.standard_normal((M, n))
# projection onto a fixed random k-dimensional subspace
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
Y = Z @ Q                                      # coordinates in the rotated basis
a = np.linalg.norm(Y[:, :k], axis=1)           # ||P Z||
b = np.linalg.norm(Y[:, k:], axis=1)           # ||(I - P) Z||

fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3), gridspec_kw={"width_ratios": [1, 1.05, 1.05]})
ax = axes[0]
ax.scatter(a, b, s=3, color=PALETTE["grey"], alpha=0.5, lw=0, rasterized=True)
th = np.linspace(0, np.pi / 2, 200)
for q, ls in [(0.1, ":"), (0.5, "--"), (0.9, ":")]:
    r = chi.ppf(q, n)
    ax.plot(r * np.cos(th), r * np.sin(th), color=PALETTE["ink"], lw=0.8, ls=ls)
i = 7
ax.plot([0, a[i], a[i], 0], [0, 0, b[i], 0], color=PALETTE["red"], lw=1.6)
ax.plot(a[i], b[i], "o", color=PALETTE["red"], ms=4)
ax.text(a[i] / 2, -0.12, r"$\Vert PZ\Vert$", color=PALETTE["red"], ha="center", va="top", fontsize=10)
ax.text(a[i] + 0.1, b[i] / 2, r"$\Vert (I-P)Z\Vert$", color=PALETTE["red"], ha="left", va="center", fontsize=10,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
ax.set_xlim(0, 5); ax.set_ylim(-0.55, 5.5); ax.set_aspect("equal")
ax.set_xlabel(r"leg in $L$ ($\chi_3$)"); ax.set_ylabel(r"leg in $L^\perp$ ($\chi_7$)")
ax.set_title(r"$n = 10$, $\dim L = 3$", loc="left")

ax = axes[1]
r = np.linspace(0, 9, 600)
for j, kk in enumerate([1, 2, 3, 10, 50]):
    ax.plot(r, chi.pdf(r, kk), color=CYCLE[j], label=rf"$\chi_{ {kk} }$")
    ax.axvline(np.sqrt(kk), color=CYCLE[j], lw=0.6, ls=":")
ax.set_xlim(0, 9); ax.set_ylim(0, 0.9)
ax.set_xlabel(r"$r$"); ax.set_title(r"Densities of $\chi_k$; dotted at $\sqrt{k}$", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.47, 1.0))

ax = axes[2]
m, S = 6, 200_000
X = rng.normal(loc=2.0, scale=3.0, size=(S, m))
Zc = (X - 2.0) / 3.0
signed_leg = np.sqrt(m) * Zc.mean(1)                      # coordinate of PZ along 1/sqrt(m)
other_leg = np.linalg.norm(Zc - Zc.mean(1, keepdims=True), axis=1)
theta = np.arctan2(other_leg, signed_leg)                 # angle between Z and the direction 1
T = np.sqrt(m - 1) / np.tan(theta)
ax.hist(T, bins=np.linspace(-6, 6, 81), density=True, color=PALETTE["grey"], alpha=0.45, edgecolor="white", lw=0.3)
x = np.linspace(-6, 6, 400)
ax.plot(x, student.pdf(x, m - 1), color=PALETTE["blue"], label=r"$t_5$ density")
ax.set_xlim(-6, 6); ax.set_ylim(0, 0.45)
ax.set_xlabel(r"$\sqrt{n-1}\,\cot\vartheta$, $n = 6$")
ax.set_title(r"The $t$ statistic as a cotangent", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "chi-triangles")
