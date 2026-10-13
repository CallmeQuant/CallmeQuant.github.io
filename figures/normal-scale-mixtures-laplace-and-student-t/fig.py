"""Normal scale mixtures: Laplace and Student t with unit variance, their mixing laws, and the
EM weights E[1/V | x] (note: normal scale mixtures)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace, t as student, expon, invgamma
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1974)
nu = 5
b = 1 / np.sqrt(2)                    # Laplace scale with unit variance
s_t = np.sqrt((nu - 2) / nu)          # t_nu scale with unit variance
M = 2_000_000

# Monte Carlo draws of sqrt(V) Z
Z = rng.standard_normal(M)
V_lap = rng.exponential(scale=2 * b ** 2, size=M)              # mean 2 b^2 = 1
V_t = (nu - 2) / rng.chisquare(nu, size=M)                     # nu s^2 / G = (nu-2)/G
X_lap, X_t = np.sqrt(V_lap) * Z, np.sqrt(V_t) * Z
edges = np.linspace(-8, 8, 65); mids = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]

fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
x = np.linspace(-8, 8, 600)
ax.plot(x, norm.pdf(x), color=PALETTE["grey"], label="normal")
ax.plot(x, laplace.pdf(x, scale=b), color=PALETTE["blue"], label="Laplace")
ax.plot(x, student.pdf(x, nu, scale=s_t), color=PALETTE["red"], label=rf"Student $t_{nu}$")
for X, col in [(X_lap, PALETTE["blue"]), (X_t, PALETTE["red"])]:
    h, _ = np.histogram(X, bins=edges)
    ok = h > 20
    ax.plot(mids[ok], h[ok] / (M * w), "o", ms=2.6, color=col, mfc="white", mew=0.8)
ax.set_yscale("log"); ax.set_ylim(1e-6, 1.2); ax.set_xlim(-8, 8)
ax.set_xlabel(r"$x$"); ax.set_title("Densities, unit variance (log scale)", loc="left")
ax.legend(loc="upper right", fontsize=8.8)

ax = axes[1]
v = np.linspace(1e-3, 4, 500)
ax.plot(v, expon.pdf(v, scale=2 * b ** 2), color=PALETTE["blue"], label=r"Laplace: $V \sim \mathrm{Exp}$, mean 1")
ax.plot(v, invgamma.pdf(v, nu / 2, scale=(nu - 2) / 2), color=PALETTE["red"],
        label=rf"$t_{nu}$: $V = 3/G$, $G \sim \chi^2_{nu}$")
ax.axvline(1, color=PALETTE["grey"], lw=1.8, label=r"normal: $V = 1$")
ax.set_xlim(0, 4); ax.set_ylim(0, 1.6)
ax.set_xlabel(r"$v$"); ax.set_title(r"Mixing densities of $V$", loc="left")
ax.legend(loc="upper right", fontsize=8.6)

ax = axes[2]
r = np.linspace(-6, 6, 801)
ax.axhline(1, color=PALETTE["grey"], label="normal")
ax.plot(r, (nu + 1) / (nu + r ** 2), color=PALETTE["red"], label=rf"$t_{nu}$")
ax.plot(r, (3 + 1) / (3 + r ** 2), color=PALETTE["gold"], label=r"$t_3$")
ax.plot(r, 2 / (1 + r ** 2), color=PALETTE["green"], label="Cauchy")
rl = r[np.abs(r) > 0.15]
ax.plot(rl[rl < 0], 1 / np.abs(rl[rl < 0]), color=PALETTE["blue"], label="Laplace")
ax.plot(rl[rl > 0], 1 / rl[rl > 0], color=PALETTE["blue"])
ax.set_xlim(-6, 6); ax.set_ylim(0, 2.6)
ax.set_xlabel(r"standardized residual $r$")
ax.set_title(r"EM weights $\mathbb{E}[V^{-1} \mid r]$", loc="left")
ax.legend(loc="upper left", fontsize=8.6, handlelength=1.4)
save(fig, __file__, "scale-mixtures")
print(np.var(X_lap), np.var(X_t))
