"""Schur complements in Gaussian conditioning and partial correlation
(note: Schur complements, block elimination and Gaussian conditioning)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, save

use_style()

# ---- Figure 1: conditional laws of X1 given X2 ---------------------------------------------
s1, s2, rho = 1.0, 1.5, 0.7
S = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
Si = np.linalg.inv(S)
g2, g1 = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-3.2, 3.2, 300))
Q = Si[0, 0] * g1**2 + 2 * Si[0, 1] * g1 * g2 + Si[1, 1] * g2**2
fig, ax = plt.subplots(figsize=(6.6, 4.1))
ax.contour(g2, g1, Q, levels=[0.5, 2, 4.5, 8], colors=PALETTE["grey"], linewidths=0.8)
b = S[0, 1] / S[1, 1]
cvar = S[0, 0] - S[0, 1] ** 2 / S[1, 1]                 # Schur complement Sigma / Sigma_22
xx = np.linspace(-4.5, 4.5, 2)
ax.plot(xx, b * xx, color=PALETTE["red"], lw=1.2, ls="--", label=r"$\mathbb{E}[X_1\mid X_2=x_2]=\Sigma_{12}\Sigma_{22}^{-1}x_2$")
t = np.linspace(-3.2, 3.2, 400)
for x2 in [-2.5, 0.0, 2.5]:
    dens = norm.pdf(t, b * x2, np.sqrt(cvar))
    ax.plot([x2, x2], [-3.2, 3.2], color=PALETTE["light"], lw=0.8, zorder=0)
    ax.fill_betweenx(t, x2, x2 + 1.2 * dens, color=PALETTE["blue"], alpha=0.18, lw=0)
    ax.plot(x2 + 1.2 * dens, t, color=PALETTE["blue"], lw=1.4)
dm = norm.pdf(t, 0, s1)
ax.plot(-4.5 + 1.2 * dm, t, color=PALETTE["gold"], lw=1.4, label=r"marginal law of $X_1$")
ax.plot([], [], color=PALETTE["blue"], lw=1.4, label=r"$X_1\mid X_2=x_2$, variance $\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$")
ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"$x_2$"); ax.set_ylabel(r"$x_1$")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.8, columnspacing=1.2)
save(fig, __file__, "gaussian-conditioning")

# ---- Figure 2: marginal versus partial correlation -----------------------------------------
rng = np.random.default_rng(7)
n = 600
# precision matrix of the chain X1 - X3 - X2: entry (1,2) is zero
Om = np.array([[2.0, 0.0, -1.2], [0.0, 2.0, -1.2], [-1.2, -1.2, 2.0]])
Sig = np.linalg.inv(Om)
X = rng.multivariate_normal(np.zeros(3), Sig, size=n)
x1, x2, x3 = X.T
r1 = x1 - (np.dot(x1, x3) / np.dot(x3, x3)) * x3
r2 = x2 - (np.dot(x2, x3) / np.dot(x3, x3)) * x3
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True, sharey=True)
sc = axes[0].scatter(x1, x2, c=x3, cmap="RdBu_r", s=9, alpha=0.8, lw=0, vmin=-3, vmax=3)
axes[0].set_title(rf"(a) $X_1$ vs $X_2$, sample corr. ${np.corrcoef(x1, x2)[0, 1]:.2f}$", loc="left")
axes[0].set_xlabel(r"$x_1$"); axes[0].set_ylabel(r"$x_2$")
axes[1].scatter(r1, r2, color=PALETTE["blue"], s=9, alpha=0.5, lw=0)
axes[1].set_title(rf"(b) residuals on $X_3$, corr. ${np.corrcoef(r1, r2)[0, 1]:.2f}$", loc="left")
axes[1].set_xlabel(r"residual of $x_1$"); axes[1].set_ylabel(r"residual of $x_2$")
for ax in axes:
    ax.set_aspect("equal")
lim = 1.05 * np.max(np.abs(np.r_[x1, x2]))
axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
cb.set_label(r"$x_3$")
save(fig, __file__, "partial-correlation", fmt="png")
print(Sig.round(3), np.corrcoef(x1, x2)[0, 1], np.corrcoef(r1, r2)[0, 1])
