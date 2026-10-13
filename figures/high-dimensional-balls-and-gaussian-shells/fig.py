"""Geometry of high-dimensional balls and Gaussians: volumes, shells, radial concentration,
near-orthogonality and distance concentration."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import chi, norm
from scipy.spatial.distance import pdist
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1961)

# ---------- Figure 1 ----------
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
ax = axes[0]
d = np.arange(1, 51)
logV = d / 2 * np.log(np.pi) - gammaln(d / 2 + 1)
ax.plot(d, np.exp(logV), "o-", ms=3, lw=1.0, color=PALETTE["blue"], label=r"$V_d$ (unit ball)")
ax.plot(d, np.exp(logV - d * np.log(2)), "o-", ms=3, lw=1.0, color=PALETTE["red"],
        label=r"$V_d / 2^d$ (ball / cube)")
ax.set_yscale("log"); ax.set_ylim(1e-28, 50); ax.set_xlim(0, 51)
ax.set_xlabel(r"dimension $d$"); ax.set_title("Volume of the unit ball", loc="left")
ax.legend(loc="lower left", fontsize=9)

ax = axes[1]
eps = np.linspace(0, 0.5, 300)
for j, dd in enumerate([2, 10, 100, 1000]):
    ax.plot(eps, 1 - (1 - eps) ** dd, color=CYCLE[j], label=rf"$d = {dd}$")
ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.03)
ax.set_xlabel(r"relative shell thickness $\varepsilon$")
ax.set_title(r"Volume fraction with $\Vert x\Vert \geq 1 - \varepsilon$", loc="left")
ax.legend(loc="lower right", fontsize=9)

ax = axes[2]
s = np.linspace(0.001, 2.2, 800)
for j, dd in enumerate([1, 3, 10, 100, 1000]):
    ax.plot(s, chi.pdf(s * np.sqrt(dd), dd) * np.sqrt(dd), color=CYCLE[j], label=rf"$d = {dd}$")
ax.set_xlim(0, 2.2); ax.set_ylim(0, 19.5)
ax.set_xlabel(r"$\Vert Z\Vert / \sqrt{d}$")
ax.set_title(r"Radius of $Z \sim N(0, I_d)$, rescaled", loc="left")
ax.legend(loc="upper right", fontsize=9)
save(fig, __file__, "volumes-and-shells")

# ---------- Figure 2 ----------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax = axes[0]
for j, dd in enumerate([3, 30, 300]):
    Z1 = rng.standard_normal((20000, dd)); Z2 = rng.standard_normal((20000, dd))
    c = (Z1 * Z2).sum(1) / np.linalg.norm(Z1, axis=1) / np.linalg.norm(Z2, axis=1)
    ax.hist(np.degrees(np.arccos(np.clip(c, -1, 1))), bins=np.linspace(0, 180, 91), density=True, color=CYCLE[j],
            alpha=0.5, label=rf"$d = {dd}$")
ax.axvline(90, color=PALETTE["ink"], lw=0.7, ls=":")
ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlabel("angle between two independent Gaussian vectors (degrees)")
ax.set_ylabel("density (per degree)")
ax.set_title("Random directions are nearly orthogonal", loc="left")
ax.legend(loc="upper left", fontsize=9)

ax = axes[1]
dims = np.unique(np.round(np.logspace(0, 4, 25)).astype(int))
contrast, lo, hi = [], [], []
for dd in dims:
    vals = []
    for rep in range(10):
        D = pdist(rng.standard_normal((100, dd)))
        vals.append((D.max() - D.min()) / D.min())
    vals = np.array(vals)
    contrast.append(np.median(vals)); lo.append(vals.min()); hi.append(vals.max())
ax.fill_between(dims, lo, hi, color=PALETTE["light"], lw=0)
ax.plot(dims, contrast, "o-", ms=3, lw=1.0, color=PALETTE["blue"], label="median of 10 replicates")
ax.plot(dims, 8 / np.sqrt(dims), color=PALETTE["ink"], lw=0.8, ls="--", label=r"slope $d^{-1/2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"dimension $d$")
ax.set_ylabel(r"$(D_{\max} - D_{\min}) / D_{\min}$")
ax.set_title("Pairwise distances of 100 Gaussian points", loc="left")
ax.legend(loc="upper right", fontsize=9)
save(fig, __file__, "angles-and-distances")
