"""Improper priors: collapse of Beta(eps, eps) posteriors when X = 0, and the posterior of log(tau)
in a normal hierarchical model under the priors 1/tau and 1."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from style import use_style, PALETTE, CYCLE, save

use_style()

# Left: Binomial(n, theta) with X = 0, Beta(eps, eps) prior -> Beta(eps, n + eps) posterior.
n = 10
t = np.linspace(0, 0.5, 600)
eps_values = [1.0, 0.3, 0.1, 0.03, 0.01]

# Right: eight groups, y_j ~ N(mu, s_j^2 + tau^2) after integrating theta_j; flat prior on mu.
y = np.array([2.1, -0.4, 1.3, 0.2, 3.0, -1.1, 0.9, 1.7])
s = np.array([1.0, 0.8, 1.5, 1.2, 1.0, 2.0, 0.9, 1.1])


def log_marg(tau):
    """log of L(tau) = int prod_j N(y_j; mu, s_j^2 + tau^2) d mu."""
    w = 1.0 / (s[None, :] ** 2 + tau[:, None] ** 2)
    W = w.sum(1)
    mu = (w * y).sum(1) / W
    q = (w * (y - mu[:, None]) ** 2).sum(1)
    J = y.size
    return -0.5 * (J - 1) * np.log(2 * np.pi) - 0.5 * np.log(W) + 0.5 * np.log(w).sum(1) - 0.5 * q


logtau = np.linspace(np.log(1e-4), np.log(1e3), 800)
lm = log_marg(np.exp(logtau))
# density of log(tau): prior 1/tau is flat in log(tau); prior 1 has density tau in log(tau)
dens_inv = np.exp(lm - lm.max())
lu = lm + logtau
dens_flat = np.exp(lu - lu.max())

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axes[0]
for k, e in enumerate(eps_values):
    ax.plot(t, beta.cdf(t, e, n + e), color=CYCLE[k], label=rf"$\varepsilon = {e:g}$")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.03)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$P(\theta \leq t \mid X = 0)$")
ax.set_title(r"Beta$(\varepsilon,\varepsilon)$ prior, $n = 10$, $X = 0$", loc="left")
ax.legend(loc="lower right")

ax = axes[1]
ax.plot(logtau / np.log(10), dens_inv, color=PALETTE["red"], label=r"$\pi(\tau) \propto 1/\tau$")
ax.plot(logtau / np.log(10), dens_flat, color=PALETTE["blue"], label=r"$\pi(\tau) \propto 1$")
ax.set_xlim(-4, 2)
ax.set_ylim(0, 1.08)
ax.set_xticks([-4, -3, -2, -1, 0, 1, 2])
ax.set_xticklabels([rf"$10^{{{k}}}$" for k in range(-4, 3)])
ax.set_xlabel(r"$\tau$ (log scale)")
ax.set_ylabel(r"density of $\log\tau$ (unnormalized)")
ax.set_title(r"Hierarchical model, $J = 8$ groups", loc="left")
ax.legend(loc="upper left")
save(fig, __file__, "improper-limits")
