"""Normalizing flows (note: densities through invertible maps). Produces two figures:
  one-dim.svg  - change of variables in one dimension: a monotone map of a uniform, a
                 diffeomorphism of a Gaussian, and a non-injective map
  coupling.png - two affine coupling layers applied to a standard Gaussian in the plane
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1505)
n = 100_000

# ------------------------------------------------------------------- one dimension
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
hist_kw = dict(bins=80, density=True, color=PALETTE["light"], edgecolor="white", lw=0.3)

# (a) X = U^2, U ~ Uniform(0, 1): p_X(x) = 1 / (2 sqrt(x)) on (0, 1)
ax = axes[0]
x = rng.uniform(size=n) ** 2
ax.hist(x, range=(0, 1), **hist_kw)
xs = np.linspace(1e-3, 1, 400)
ax.plot(xs, 1 / (2 * np.sqrt(xs)), color=PALETTE["blue"], label=r"$1/(2\sqrt{x})$")
ax.set_ylim(0, 5); ax.set_xlim(0, 1)
ax.set_title(r"(a) $U \sim \mathrm{Unif}(0,1)$, $X = U^2$", loc="left")
ax.legend(loc="upper right")

# (b) X = T(U) = U + 2 tanh(2U), U ~ N(0, 1): parametric curve (T(u), phi(u) / T'(u))
ax = axes[1]
T = lambda u: u + 2 * np.tanh(2 * u)
dT = lambda u: 1 + 4 / np.cosh(2 * u) ** 2
x = T(rng.standard_normal(n))
ax.hist(x, range=(-6, 6), **hist_kw)
us = np.linspace(-6, 6, 800)
ax.plot(T(us), norm.pdf(us) / dT(us), color=PALETTE["blue"], label=r"$\varphi(u)/T'(u)$")
ax.plot(us, norm.pdf(us), color=PALETTE["grey"], ls="--", lw=1.2, label=r"base $\varphi$")
ax.set_xlim(-6, 6); ax.set_ylim(0, 0.55)
ax.set_title(r"(b) $X = U + 2\tanh(2U)$, $U \sim \mathcal{N}(0,1)$", loc="left")
ax.legend(loc="upper right", fontsize=8.5)

# (c) X = U^2, U ~ N(0, 1): two preimages; the one-branch formula is off by a factor 2
ax = axes[2]
x = rng.standard_normal(n) ** 2
x = x[x <= 5]          # normalize by all n draws: about 2.5% of the mass lies beyond 5
ax.hist(x, bins=80, range=(0, 5), weights=np.full(x.size, 80 / (5 * n)),
        color=PALETTE["light"], edgecolor="white", lw=0.3)
xs = np.linspace(0.02, 5, 400)
ax.plot(xs, chi2.pdf(xs, 1), color=PALETTE["blue"], label="sum over both preimages")
ax.plot(xs, norm.pdf(np.sqrt(xs)) / (2 * np.sqrt(xs)), color=PALETTE["red"], ls="--",
        label=r"branch $u = +\sqrt{x}$ only")
ax.set_xlim(0, 5); ax.set_ylim(0, 1.5)
ax.set_title(r"(c) $X = U^2$, $U \sim \mathcal{N}(0,1)$", loc="left")
ax.legend(loc="upper right", fontsize=8.5)
for ax in axes:
    ax.set_xlabel(r"$x$")
axes[0].set_ylabel("density")
save(fig, __file__, "one-dim")

# ------------------------------------------------------------------- coupling layers
s1 = lambda a: 0.5 * np.tanh(a)            # layer 1 updates coordinate 2 given coordinate 1
t1 = lambda a: 0.6 * a ** 2 - 0.6
s2 = lambda b: -0.3 * np.tanh(b)            # layer 2 updates coordinate 1 given coordinate 2
t2 = lambda b: 0.9 * np.sin(1.2 * b)


def layer1(u1, u2):
    return u1, u2 * np.exp(s1(u1)) + t1(u1)


def layer2(x1, x2):
    return x1 * np.exp(s2(x2)) + t2(x2), x2


def logp_stage(stage, y1, y2):
    """Exact log density after `stage` layers (0, 1 or 2), by inverting the layers."""
    ld = 0.0
    if stage >= 2:
        y1 = (y1 - t2(y2)) * np.exp(-s2(y2)); ld = ld - s2(y2)
    if stage >= 1:
        ld = ld - s1(y1)
        y2 = (y2 - t1(y1)) * np.exp(-s1(y1))
    return norm.logpdf(y1) + norm.logpdf(y2) + ld


u = rng.standard_normal((2000, 2))
pts = [(u[:, 0], u[:, 1])]
pts.append(layer1(*pts[0]))
pts.append(layer2(*pts[1]))

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharex=True, sharey=True)
g = np.linspace(-4, 4, 301)
G1, G2 = np.meshgrid(g, g)
lines = np.linspace(-2, 2, 5)
tt = np.linspace(-2.6, 2.6, 200)
titles = ["base $\\mathcal{N}(0, I_2)$", "after coupling layer 1", "after coupling layer 2"]
for stage, ax in enumerate(axes):
    for c in lines:                       # images of the base grid lines u1 = c and u2 = c
        for a, b in ((np.full_like(tt, c), tt), (tt, np.full_like(tt, c))):
            z = (a, b)
            if stage >= 1: z = layer1(*z)
            if stage >= 2: z = layer2(*z)
            ax.plot(*z, color=PALETTE["grey"], lw=0.6, alpha=0.8)
    p1, p2 = pts[stage]
    ax.scatter(p1, p2, s=2, color=PALETTE["blue"], alpha=0.35, lw=0)
    L = logp_stage(stage, G1, G2)
    ax.contour(g, g, np.exp(L), levels=[0.01, 0.04, 0.1], colors=PALETTE["red"], linewidths=0.9)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 4.1)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_title(titles[stage], loc="left")
    ax.set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
save(fig, __file__, "coupling", fmt="png")

# check: exact density integrates to 1 on the grid and matches samples
for stage in range(3):
    L = logp_stage(stage, G1, G2)
    print(stage, "mass on grid", np.exp(L).sum() * (g[1] - g[0]) ** 2)
