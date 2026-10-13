"""Fisher geometry of the normal family N(mu, sigma^2): KL balls in the (mu, sigma) plane, and
gradient descent versus natural gradient descent for maximum likelihood
(note: natural gradient and the Fisher metric)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()


def kl(m0, s0, m, s):          # KL( N(m0, s0^2) || N(m, s^2) )
    return np.log(s / s0) + (s0**2 + (m0 - m)**2) / (2 * s**2) - 0.5


xbar, s2 = 1.0, 1.0            # sufficient statistics of the data: mean and mean squared deviation
def loss(m, s): return np.log(s) + (s2 + (xbar - m)**2) / (2 * s**2)
def grad(m, s): return np.array([-(xbar - m) / s**2, 1 / s - (s2 + (xbar - m)**2) / s**3])
def fisher_inv(m, s): return np.diag([s**2, s**2 / 2])

start, eta, K = np.array([-2.0, 0.5]), 0.05, 100
paths = {}
for name in ("gradient descent", "natural gradient descent"):
    th = start.copy(); P = [th.copy()]
    for _ in range(K):
        d = grad(*th) if name == "gradient descent" else fisher_inv(*th) @ grad(*th)
        th = th - eta * d; P.append(th.copy())
    paths[name] = np.array(P)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
ax = axes[0]
mm, ss = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(0.02, 3.2, 500))
for s0 in (0.25, 0.5, 1.0, 2.0):
    for m0 in (-2, 0, 2):
        ax.contour(mm, ss, kl(m0, s0, mm, ss), levels=[0.05], colors=PALETTE["blue"], linewidths=1.0)
        ax.plot(m0, s0, ".", color=PALETTE["ink"], ms=3)
ax.set_xlim(-3, 3); ax.set_ylim(0, 3.2)
ax.set_aspect("equal")
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\sigma$")
ax.set_title(r"(a) Balls $\{\theta: \mathrm{KL}(p_{\theta_0}\Vert p_\theta)\leq 0.05\}$", loc="left")

ax = axes[1]
mm, ss = np.meshgrid(np.linspace(-2.6, 2.0, 400), np.linspace(0.05, 4.8, 400))
Lstar = loss(xbar, np.sqrt(s2))
ax.contour(mm, ss, loss(mm, ss) - Lstar, levels=np.geomspace(0.01, 8, 12), colors=PALETTE["grey"], linewidths=0.6)
cols = {"gradient descent": PALETTE["red"], "natural gradient descent": PALETTE["blue"]}
for name, P in paths.items():
    ax.plot(P[:, 0], P[:, 1], "-o", color=cols[name], ms=2.2, lw=1.1,
            label=("GD" if name == "gradient descent" else "natural GD") + f": gap {loss(*P[-1]) - Lstar:.1e}")
ax.plot(*start, "s", color=PALETTE["ink"], ms=5)
ax.plot(xbar, np.sqrt(s2), "*", color=PALETTE["ink"], ms=10)
ax.set_xlim(-2.6, 2.0); ax.set_ylim(0, 4.8)
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\sigma$")
ax.set_title(rf"(b) Fitting $N(\mu,\sigma^2)$, step $\eta={eta}$, {K} steps", loc="left")
ax.legend(loc="upper right", fontsize=8.5, framealpha=1, frameon=True, edgecolor="none")
save(fig, __file__, "fisher-geometry")
