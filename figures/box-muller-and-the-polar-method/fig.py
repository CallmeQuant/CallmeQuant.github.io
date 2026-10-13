"""Box-Muller map and Marsaglia's polar method (note: from two uniforms to two Gaussians)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1958)


def box_muller(u1, u2):
    r = np.sqrt(-2.0 * np.log(u1))
    return r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)


# ---------- Figure 1: the Box-Muller map acting on a grid of the unit square ----------
levels_u1 = [0.02, 0.1, 0.3, 0.6, 0.9]          # vertical lines -> circles
levels_u2 = np.arange(0, 1, 1 / 8)              # horizontal lines -> rays
cols = [CYCLE[k] for k in range(len(levels_u1))]
n_pts = 1500
u1, u2 = 1 - rng.random(n_pts), rng.random(n_pts)   # u1 in (0, 1]
z1, z2 = box_muller(u1, u2)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw={"width_ratios": [1, 1.05]})
ax = axes[0]
ax.scatter(u1, u2, s=3, color=PALETTE["grey"], alpha=0.55, lw=0)
for c, a in zip(cols, levels_u1):
    ax.plot([a, a], [0, 1], color=c, lw=1.6)
for b in levels_u2:
    ax.plot([0, 1], [b, b], color=PALETTE["ink"], lw=0.6, alpha=0.6)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel(r"$u_1$"); ax.set_ylabel(r"$u_2$")
ax.set_title(r"Uniform pairs $(U_1, U_2)$", loc="left")
ax.grid(False)

ax = axes[1]
ax.scatter(z1, z2, s=3, color=PALETTE["grey"], alpha=0.55, lw=0)
t = np.linspace(0, 2 * np.pi, 400)
for c, a in zip(cols, levels_u1):
    r = np.sqrt(-2 * np.log(a))
    ax.plot(r * np.cos(t), r * np.sin(t), color=c, lw=1.6, label=rf"$u_1 = {a}$")
for b in levels_u2:
    ax.plot([0, 3.3 * np.cos(2 * np.pi * b)], [0, 3.3 * np.sin(2 * np.pi * b)],
            color=PALETTE["ink"], lw=0.6, alpha=0.6)
ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
ax.set_title(r"Image $(Z_1, Z_2)$", loc="left")
ax.grid(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.2)
save(fig, __file__, "box-muller-map")

# ---------- Figure 2: polar method (rejection step and output) ----------
m = 1200
v = rng.uniform(-1, 1, size=(m, 2))
s = (v ** 2).sum(1)
acc = (s > 0) & (s < 1)

N = 200_000
w = rng.uniform(-1, 1, size=(int(1.4 * N), 2))
sw = (w ** 2).sum(1)
w, sw = w[(sw > 0) & (sw < 1)][:N], sw[(sw > 0) & (sw < 1)][:N]
z = w * np.sqrt(-2 * np.log(sw) / sw)[:, None]

fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
ax = axes[0]
ax.scatter(*v[acc].T, s=3, color=PALETTE["blue"], lw=0, label="accepted")
ax.scatter(*v[~acc].T, s=3, color=PALETTE["red"], lw=0, label="rejected")
ax.plot(np.cos(t), np.sin(t), color=PALETTE["ink"], lw=0.8)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r"$v_1$"); ax.set_ylabel(r"$v_2$")
ax.set_title(f"Accept rate {acc.mean():.3f}" + r" ($\pi/4 \approx 0.785$)", loc="left", fontsize=10)
ax.grid(False)

ax = axes[1]
ax.hist(sw, bins=40, range=(0, 1), density=True, color=PALETTE["grey"], alpha=0.45, edgecolor="white", lw=0.4)
ax.axhline(1.0, color=PALETTE["ink"], lw=1.2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.35)
ax.set_xlabel(r"$S = V_1^2 + V_2^2$ (accepted)"); ax.set_ylabel("density")
ax.set_title(r"$S \sim \mathrm{Uniform}(0,1)$", loc="left")

ax = axes[2]
x = np.linspace(-4, 4, 400)
ax.hist(z[:, 0], bins=60, range=(-4, 4), density=True, color=PALETTE["grey"], alpha=0.45, edgecolor="white", lw=0.4)
ax.plot(x, norm.pdf(x), color=PALETTE["blue"], lw=1.6)
ax.set_xlim(-4, 4); ax.set_ylim(0, 0.47)
ax.set_xlabel(r"$Z_1$"); ax.set_ylabel("density")
ax.set_title(r"Output vs. $N(0,1)$ density", loc="left")
save(fig, __file__, "polar-method")
