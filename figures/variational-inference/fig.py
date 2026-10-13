"""Coordinate ascent variational inference with a mean-field Gaussian family for a bivariate
Gaussian target (note: variational inference and the evidence lower bound)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from style import use_style, PALETTE, CYCLE, save

use_style()
mu = np.array([1.0, 0.5])
m0 = np.array([-1.5, 2.5])          # initial means; variances start at their optimal values 1 - rho^2


def cavi(rho, sweeps):
    Lam = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    m = m0.copy()
    path = [m.copy()]
    for _ in range(sweeps):
        m[0] = mu[0] - Lam[0, 1] / Lam[0, 0] * (m[1] - mu[1]); path.append(m.copy())
        m[1] = mu[1] - Lam[1, 0] / Lam[1, 1] * (m[0] - mu[0]); path.append(m.copy())
    return np.array(path), Lam


def kl_meanfield(m, rho):
    """KL(q || p) for q = N(m, diag(1 - rho^2)) and p = N(mu, [[1, rho], [rho, 1]])."""
    Lam = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    d = m - mu
    return 0.5 * d @ Lam @ d - 0.5 * np.log(1 - rho**2)


fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})

# (a) target, optimal mean-field approximation, CAVI path for rho = 0.8
rho = 0.8
ax = axes[0]
Sigma = np.array([[1.0, rho], [rho, 1.0]])
w, V = np.linalg.eigh(Sigma)
ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
for r in (1, 2):
    ax.add_patch(Ellipse(mu, 2 * r * np.sqrt(w[1]), 2 * r * np.sqrt(w[0]), angle=ang, fill=False,
                         color=PALETTE["ink"], lw=1.3, label=r"target $p$" if r == 1 else None))
    s = np.sqrt(1 - rho**2)
    ax.add_patch(Ellipse(mu, 2 * r * s, 2 * r * s, fill=False, color=PALETTE["red"], lw=1.3,
                         ls=(0, (4, 2)), label=r"optimal $q^\ast$" if r == 1 else None))
path, _ = cavi(rho, 8)
ax.plot(path[:, 0], path[:, 1], color=PALETTE["blue"], lw=1.1, marker="o", ms=3.2, label="CAVI means")
ax.plot(*m0, "o", color=PALETTE["blue"], ms=5)
ax.annotate(r"start", xy=m0, xytext=(m0[0] + 0.2, m0[1] + 0.15), fontsize=9.5, color=PALETTE["blue"])
ax.set_xlim(-2.4, 3.6); ax.set_ylim(-2.2, 3.2)
ax.set_aspect("equal")
ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
ax.set_title(r"(a) $\rho = 0.8$: contours at radii 1 and 2", loc="left")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=PALETTE["ink"], lw=1.3, label=r"target $p$"),
           Line2D([], [], color=PALETTE["red"], lw=1.3, ls=(0, (4, 2)), label=r"optimal $q^\ast$"),
           Line2D([], [], color=PALETTE["blue"], lw=1.1, marker="o", ms=3.2, label="CAVI means")]
ax.legend(handles=handles, loc="lower right", fontsize=9)

# (b) KL(q_t || p) after each half-sweep, against the floor -log(1 - rho^2)/2
ax = axes[1]
for k, rho in enumerate([0.5, 0.8, 0.95]):
    path, _ = cavi(rho, 30)
    kl = np.array([kl_meanfield(m, rho) for m in path[1:]])
    t = np.arange(1, kl.size + 1) / 2
    floor = -0.5 * np.log(1 - rho**2)
    ax.plot(t, kl, color=CYCLE[k], lw=1.6, label=rf"$\rho = {rho}$")
    ax.axhline(floor, color=CYCLE[k], lw=0.9, ls=(0, (2, 2)))
ax.set_yscale("log")
ax.set_xlim(0, 15)
ax.set_ylim(0.1, 6)
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("sweep")
ax.set_ylabel(r"$\mathrm{KL}(q_t \,\Vert\, p) = \log p(x) - \mathrm{ELBO}(q_t)$")
ax.set_title(r"(b) gap after each coordinate update", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "cavi-gaussian")

for rho in (0.5, 0.8, 0.95):
    print(rho, "floor", -0.5 * np.log(1 - rho**2))
