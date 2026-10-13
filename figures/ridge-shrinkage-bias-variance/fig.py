"""Ridge shrinkage: geometry of the ridge path for two correlated covariates, and shrinkage
factors with the bias-variance decomposition of the mean squared error
(note: where ridge regression shrinks)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()

# ---- Figure 1: two correlated covariates, X^T X = [[1, rho], [rho, 1]] -----------------
rho = 0.9
G = np.array([[1.0, rho], [rho, 1.0]])
b_ols = np.array([2.0, -0.3])
lams = np.concatenate([[0], np.logspace(-3, 2.5, 400)])
path = np.array([np.linalg.solve(G + l * np.eye(2), G @ b_ols) for l in lams])

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
g1, g2 = np.meshgrid(np.linspace(-1.2, 3.0, 300), np.linspace(-1.2, 2.2, 300))
D = np.stack([g1 - b_ols[0], g2 - b_ols[1]], -1)
rss = np.einsum("...i,ij,...j->...", D, G, D)          # RSS minus its minimum
ax.contour(g1, g2, rss, levels=[0.05, 0.2, 0.5, 1.0, 1.6, 2.4], colors=PALETTE["grey"], linewidths=0.8)
for rad in [0.5, 1.0, 1.5]:
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rad * np.cos(t), rad * np.sin(t), color=PALETTE["light"], lw=1.0, zorder=0)
ax.plot(path[:, 0], path[:, 1], color=PALETTE["blue"], lw=2)
for l in [0.1, 1.0, 5.0]:
    q = np.linalg.solve(G + l * np.eye(2), G @ b_ols)
    ax.plot(*q, "o", color=PALETTE["blue"], ms=4)
    ax.annotate(rf"$\lambda={l:g}$", q, xytext=(4, -13), textcoords="offset points", fontsize=9,
                color=PALETTE["blue"], bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
ax.plot(*b_ols, "o", color=PALETTE["red"], ms=6)
ax.annotate(r"$\hat\beta^{\mathrm{OLS}}$", b_ols, xytext=(7, -3), textcoords="offset points", color=PALETTE["red"])
ax.plot(0, 0, "+", color=PALETTE["ink"], ms=8)
for v, lab in [(np.array([1, 1]) / np.sqrt(2), r"$v_1$"), (np.array([1, -1]) / np.sqrt(2), r"$v_2$")]:
    ax.annotate("", xy=0.9 * v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.2))
    ax.annotate(lab, 0.9 * v, xytext=(3, 3 if lab == r"$v_1$" else -12), textcoords="offset points", color=PALETTE["green"])
ax.set_aspect("equal")
ax.set_xlim(-1.2, 3.0); ax.set_ylim(-1.2, 2.2)
ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$")
ax.set_title("(a) Ridge path in the parameter plane", loc="left")

ax = axes[1]
lp = np.logspace(-3, 2.5, 400)
pp = np.array([np.linalg.solve(G + l * np.eye(2), G @ b_ols) for l in lp])
ax.plot(lp, pp[:, 0], color=CYCLE[0], label=r"$\hat\beta_{\lambda,1}$")
ax.plot(lp, pp[:, 1], color=CYCLE[1], label=r"$\hat\beta_{\lambda,2}$")
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (log scale)")
ax.set_title("(b) Coefficients along the path", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "ridge-geometry")

# ---- Figure 2: shrinkage factors and MSE decomposition ----------------------------------
d2 = np.array([50.0, 20.0, 8.0, 3.0, 1.0, 0.3])          # squared singular values
alpha = np.array([1.0, -0.8, 0.6, 0.5, -0.4, 0.3])       # coefficients in the V basis
sigma2 = 1.0
lam = np.logspace(-3, 3, 400)
s = d2[None, :] / (d2[None, :] + lam[:, None])
bias2 = ((1 - s) ** 2 * alpha**2).sum(1)
var = (s**2 * sigma2 / d2).sum(1)
mse = bias2 + var
mse_ols = (sigma2 / d2).sum()
lam_hk = 2 * sigma2 / np.max(alpha**2)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
for j in range(len(d2)):
    ax.plot(lam, s[:, j], color=CYCLE[j % len(CYCLE)], label=rf"$d_{j+1}^2={d2[j]:g}$")
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$\lambda$ (log scale)")
ax.set_ylabel(r"$s_j(\lambda)=d_j^2/(d_j^2+\lambda)$")
ax.set_title("(a) Shrinkage factors", loc="left")
ax.legend(loc="lower left", fontsize=8.5)

ax = axes[1]
ax.axvspan(lam[0], lam_hk, color=PALETTE["light"], alpha=0.7, lw=0)
ax.plot(lam, mse, color=PALETTE["ink"], label="MSE")
ax.plot(lam, bias2, color=PALETTE["red"], ls="--", label="squared bias")
ax.plot(lam, var, color=PALETTE["blue"], ls="--", label="variance")
ax.axhline(mse_ols, color=PALETTE["grey"], lw=1, ls=":")
ax.text(lam[-1], mse_ols * 1.03, "OLS", ha="right", va="bottom", color=PALETTE["grey"], fontsize=9)
k = mse.argmin()
ax.plot(lam[k], mse[k], "o", color=PALETTE["ink"], ms=4)
ax.set_xscale("log")
ax.set_ylim(0, 5.6)
ax.set_xlabel(r"$\lambda$ (log scale)")
ax.set_title("(b) Mean squared error of $\\hat\\beta_\\lambda$", loc="left")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
save(fig, __file__, "shrinkage-mse")
print(mse_ols, lam[k], mse[k], lam_hk)
