"""EM for the symmetric two-component mixture 0.5 N(theta, 1) + 0.5 N(-theta, 1): lower bounds
F(q_t, theta) touching the log-likelihood, and linear convergence at the missing-information rate
(note: EM as coordinate ascent on the evidence lower bound)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1977)
n = 200


def simulate(theta_star):
    z = rng.choice([-1.0, 1.0], size=n)
    return z * theta_star + rng.standard_normal(n)


def loglik(theta, x):
    th = np.atleast_1d(theta)[:, None]
    return np.sum(np.logaddexp(-0.5 * (x - th) ** 2, -0.5 * (x + th) ** 2) - np.log(2) - 0.5 * np.log(2 * np.pi), axis=1)


def lower_bound(theta, x, theta_t):
    """F(q_t, theta) with q_t = p(z | x, theta_t): E_q log p(x, z | theta) + H(q)."""
    r = 1 / (1 + np.exp(-2 * theta_t * x))                       # P(z_i = +1 | x_i, theta_t)
    th = np.atleast_1d(theta)[:, None]
    ecomp = r * (-0.5 * (x - th) ** 2) + (1 - r) * (-0.5 * (x + th) ** 2) - np.log(2) - 0.5 * np.log(2 * np.pi)
    ent = -(r * np.log(r) + (1 - r) * np.log1p(-r))
    return np.sum(ecomp + ent, axis=1)


def em_map(theta, x):
    return np.mean(x * np.tanh(theta * x))


# ---------------- panel (a): minorization -------------------------------------------------------
x = simulate(1.0)
theta = 0.15
iters = [theta]
for _ in range(4):
    theta = em_map(theta, x)
    iters.append(theta)
grid = np.linspace(-0.2, 1.8, 400)
ll = loglik(grid, x)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axes[0]
ax.plot(grid, ll, color=PALETTE["ink"], lw=1.8, label=r"$\ell(\theta)$")
for t in range(4):
    lb = lower_bound(grid, x, iters[t])
    ax.plot(grid, lb, color=CYCLE[t], lw=1.2, ls="--", label=rf"$F(q_{t+1},\theta)$")
    ax.plot([iters[t]], loglik(iters[t], x), "o", color=CYCLE[t], ms=4)
    tn = iters[t + 1]
    ax.plot([tn], lower_bound(tn, x, iters[t]), "^", color=CYCLE[t], ms=4.5)
ax.set_xlim(-0.2, 1.8)
lo = ll.max()
ax.set_ylim(lo - 70, lo + 8)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("log-likelihood and lower bounds")
ax.set_title(r"Bounds touching $\ell$ at $\theta_t$")
ax.legend(loc="upper left", fontsize=8.6)

# ---------------- panel (b): convergence rates --------------------------------------------------
ax = axes[1]
for k, ts in enumerate([0.6, 1.0, 2.0]):
    xs = simulate(ts)
    th = 0.5 * ts if ts > 0.7 else 1.2
    hist = [th]
    for _ in range(400):
        th = em_map(th, xs)
        hist.append(th)
    hat = hist[-1]
    rate = np.mean(xs ** 2 / np.cosh(hat * xs) ** 2)          # M'(theta_hat) = mean missing information
    err = np.abs(np.array(hist[:60]) - hat)
    T = np.arange(err.size)
    keep = err > 1e-13
    ax.semilogy(T[keep], err[keep], color=CYCLE[k], marker="o", ms=3, lw=0,
                label=rf"$\theta^\ast={ts}$: rate $\hat\rho={rate:.2f}$")
    ax.semilogy(T, err[3] * rate ** (T - 3), color=CYCLE[k], ls=":", lw=1.1)
    print(f"theta*={ts}: theta_hat={hat:.4f}, rate={rate:.4f}")
ax.set_xlim(0, 59)
ax.set_ylim(1e-12, 3)
ax.set_xlabel(r"iteration $t$")
ax.set_ylabel(r"$|\theta_t-\hat\theta|$")
ax.set_title("Linear convergence")
ax.legend(loc="upper right", fontsize=8.6)
save(fig, __file__, "em-bounds-and-rates")
