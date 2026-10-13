"""Leapfrog trajectories, energy error and integration time for HMC on Gaussian targets
(note: Hamiltonian Monte Carlo)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()

# Target N(0, Sigma), U(q) = q' Sigma^{-1} q / 2, identity mass matrix.
Sigma = np.array([[1.0, 0.95], [0.95, 1.0]])
Prec = np.linalg.inv(Sigma)
sig_min = np.sqrt(np.linalg.eigvalsh(Sigma).min())          # stability needs eps < 2 * sig_min
H = lambda q, p: 0.5 * np.einsum("...i,ij,...j->...", q, Prec, q) + 0.5 * (p**2).sum(-1)


def leapfrog(q, p, eps, L):
    """Return the whole leapfrog path (positions and momenta) of L steps; q, p may be batched."""
    qs, ps = [q], [p]
    for _ in range(L):
        p = p - 0.5 * eps * q @ Prec
        q = q + eps * p
        p = p - 0.5 * eps * q @ Prec
        qs.append(q); ps.append(p)
    return np.array(qs), np.array(ps)


# ---- Figure 1: trajectories and energy error for three step sizes
q0, p0 = np.array([-1.2, -0.9]), np.array([1.4, -0.6])
cases = [(0.05, 64, PALETTE["blue"]), (0.40, 8, PALETTE["green"]), (0.46, 3, PALETTE["red"])]
g = np.linspace(-2.6, 2.6, 300)
X, Y = np.meshgrid(g, g)
D = np.exp(-H(np.stack([X, Y], -1), np.zeros((300, 300, 2))))

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
ax.contour(X, Y, D, levels=[0.05, 0.25, 0.5, 0.8], colors=PALETTE["grey"], linewidths=0.8)
for eps, L, col in cases:
    qs, _ = leapfrog(q0, p0, eps, L)
    ax.plot(qs[:, 0], qs[:, 1], "-o", color=col, ms=2.2 if L > 20 else 3.2, lw=1.1,
            label=rf"$\varepsilon = {eps}$")
ax.plot(*q0, "s", color=PALETTE["ink"], ms=5)
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal")
ax.set_xlabel(r"$q_1$"); ax.set_ylabel(r"$q_2$")
ax.set_title("Leapfrog paths in position space", loc="left")
ax.legend(loc="upper left", handlelength=1.2)

ax = axes[1]
for eps, col in [(0.05, PALETTE["blue"]), (0.40, PALETTE["green"]), (0.46, PALETTE["red"])]:
    L = int(round(6 / eps))
    qs, ps = leapfrog(q0, p0, eps, L)
    err = np.abs(H(qs, ps) - H(q0, p0))
    t = eps * np.arange(L + 1)
    ax.plot(t[1:], err[1:], color=col, lw=1.3, label=rf"$\varepsilon = {eps}$")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1e4); ax.set_xlim(0, 6)
ax.set_xlabel(r"integration time $k\varepsilon$")
ax.set_ylabel(r"$\vert H(z_k) - H(z_0)\vert$")
ax.set_title(rf"Energy error (stability limit $2\sigma_{{\min}} = {2 * sig_min:.3f}$)", loc="left")
save(fig, __file__, "leapfrog-energy")

# ---- Figure 2: lag-one autocorrelation as a function of integration time T
rng = np.random.default_rng(11)
N = 20_000


def hmc_step_from_stationarity(chol, eps, T, rng):
    """Draw q0 ~ N(0, Sigma), apply one HMC transition, return (q0, q1)."""
    d = chol.shape[0]
    P = np.linalg.inv(chol @ chol.T)
    q0 = rng.standard_normal((N, d)) @ chol.T
    p = rng.standard_normal((N, d))
    q, L = q0.copy(), max(1, int(round(T / eps)))
    eps = T / L
    h0 = 0.5 * np.einsum("ni,ij,nj->n", q, P, q) + 0.5 * (p**2).sum(1)
    for _ in range(L):
        p = p - 0.5 * eps * q @ P; q = q + eps * p; p = p - 0.5 * eps * q @ P
    h1 = 0.5 * np.einsum("ni,ij,nj->n", q, P, q) + 0.5 * (p**2).sum(1)
    acc = np.log(rng.random(N)) < h0 - h1
    return q0, np.where(acc[:, None], q, q0)


corr = lambda a, b: np.corrcoef(a, b)[0, 1]
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))

# (a) one dimension, sigma = 1
T = np.linspace(0, 2 * np.pi, 400)
Te = np.linspace(0.15, 2 * np.pi, 24)
ax = axes[0]
ax.plot(T, np.cos(T), color=PALETTE["blue"], label=r"$h(q) = q$: $\cos T$")
ax.plot(T, np.cos(T) ** 2, color=PALETTE["red"], label=r"$h(q) = q^2$: $\cos^2 T$")
emp = [hmc_step_from_stationarity(np.eye(1), 0.05, t, rng) for t in Te]
ax.plot(Te, [corr(a[:, 0], b[:, 0]) for a, b in emp], "o", color=PALETTE["blue"], ms=3.5)
ax.plot(Te, [corr(a[:, 0] ** 2, b[:, 0] ** 2) for a, b in emp], "o", color=PALETTE["red"], ms=3.5)
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_ylim(-1.15, 1.6)
ax.set_xlabel(r"integration time $T$"); ax.set_ylabel("lag-one autocorrelation")
ax.set_title(r"$N(0, 1)$", loc="left")
ax.legend(loc="upper center", ncol=1)

# (b) the correlated two-dimensional target, along its principal axes
w, V = np.linalg.eigh(Sigma)
sd = np.sqrt(w)
chol = np.linalg.cholesky(Sigma)
T = np.linspace(0, 2 * np.pi * sd[1], 600)
Te = np.linspace(0.1, 2 * np.pi * sd[1], 26)
ax = axes[1]
emp = [hmc_step_from_stationarity(chol, 0.1 * sd[0], t, rng) for t in Te]
for j, col, name in [(1, PALETTE["blue"], "wide"), (0, PALETTE["red"], "narrow")]:
    ax.plot(T, np.cos(T / sd[j]), color=col, lw=1.4,
            label=rf"{name} axis, $\sigma = {sd[j]:.2f}$")
    ax.plot(Te, [corr(a @ V[:, j], b @ V[:, j]) for a, b in emp], "o", color=col, ms=3.5)
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.set_ylim(-1.15, 1.6)
ax.set_xlabel(r"integration time $T$")
ax.set_title(r"$N(0, \Sigma)$, $\rho = 0.95$: principal components", loc="left")
ax.legend(loc="upper center")
save(fig, __file__, "integration-time")
