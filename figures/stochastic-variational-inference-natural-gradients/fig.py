"""Stochastic variational inference for a Gaussian mixture with unknown means: data and the path of the
variational means, and the ELBO against the number of data points processed for batch coordinate ascent
and SVI (note: stochastic variational inference)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2013)
N, K, d, SIG0 = 20_000, 4, 2, 10.0              # data size, components, dimension, prior sd of the means
centres = np.array([[-3.0, -1.0], [0.5, 3.0], [3.0, -2.0], [-0.5, 0.0]])
labels = rng.integers(K, size=N)
X = centres[labels] + rng.standard_normal((N, d))


def expected_loglik(X, m, s2):
    """E_q log N(x_i; mu_k, I) for every i, k (N x K)."""
    sq = (X ** 2).sum(1)[:, None] - 2 * X @ m.T + (m ** 2).sum(1)[None, :]
    return -0.5 * d * np.log(2 * np.pi) - 0.5 * (sq + d * s2[None, :])


def elbo_per_point(m, s2):
    """ELBO maximized over the local factors, divided by N."""
    local = logsumexp(np.log(1 / K) + expected_loglik(X, m, s2), axis=1).sum()
    prior = (-0.5 * d * np.log(2 * np.pi * SIG0 ** 2) - (np.sum(m ** 2, 1) + d * s2) / (2 * SIG0 ** 2)).sum()
    ent = (0.5 * d * np.log(2 * np.pi * np.e * s2)).sum()
    return (local + prior + ent) / N


def nat_from_suffstats(sx, sn, scale):
    """Natural parameters (eta1 = m / s2, eta2 = -1 / (2 s2)) of the CAVI target alpha + scale * stats."""
    eta1 = scale * sx
    eta2 = -1 / (2 * SIG0 ** 2) - 0.5 * scale * sn
    return eta1, eta2


def to_mean(eta1, eta2):
    s2 = -1 / (2 * eta2)
    return eta1 * s2[:, None], s2


def local_stats(Xb, m, s2):
    lp = np.log(1 / K) + expected_loglik(Xb, m, s2)
    phi = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    return phi.T @ Xb, phi.sum(0)            # sum_i phi_ik x_i, sum_i phi_ik


init_m = rng.normal(0, 0.5, size=(K, d))
init_s2 = np.ones(K)

# batch coordinate ascent (CAVI): one iteration = one pass over the data
m, s2 = init_m.copy(), init_s2.copy()
cavi_x, cavi_y = [0], [elbo_per_point(m, s2)]
for it in range(40):
    sx, sn = local_stats(X, m, s2)
    m, s2 = to_mean(*nat_from_suffstats(sx, sn, 1.0))
    cavi_x.append((it + 1) * N)
    cavi_y.append(elbo_per_point(m, s2))


def svi(B, kappa=0.7, tau=1.0, steps=None, seed=0, n_record=60):
    r = np.random.default_rng(seed)
    m, s2 = init_m.copy(), init_s2.copy()
    eta1, eta2 = m / s2[:, None], -1 / (2 * s2)
    xs, ys, path = [0], [elbo_per_point(m, s2)], [m.copy()]
    record = set(np.unique(np.geomspace(1, steps, n_record).astype(int)))
    for t in range(steps):
        idx = r.integers(N, size=B)
        sx, sn = local_stats(X[idx], m, s2)
        h1, h2 = nat_from_suffstats(sx, sn, N / B)
        rho = (t + tau) ** (-kappa)
        eta1 = (1 - rho) * eta1 + rho * h1
        eta2 = (1 - rho) * eta2 + rho * h2
        m, s2 = to_mean(eta1, eta2)
        path.append(m.copy())
        if (t + 1) in record:
            xs.append((t + 1) * B)
            ys.append(elbo_per_point(m, s2))
    return np.array(xs), np.array(ys), np.array(path)


runs = {B: svi(B, steps=int(12 * N / B)) for B in (10, 100)}
for B, (xs, ys, _) in runs.items():
    print(f"SVI B={B}: ELBO/N after 1 pass {ys[np.searchsorted(xs, N)]:.4f}, final {ys[-1]:.4f}")
print("CAVI per pass:", np.round(cavi_y[:13], 4))
best = max(cavi_y)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
sub = rng.choice(N, 2500, replace=False)
ax.scatter(X[sub, 0], X[sub, 1], s=2, color=PALETTE["grey"], alpha=0.35, lw=0, rasterized=True)
path = runs[10][2]
for k in range(K):
    ax.plot(path[:, k, 0], path[:, k, 1], color=CYCLE[k], lw=1.0)
    ax.plot(path[0, k, 0], path[0, k, 1], "o", ms=4, mfc="white", color=CYCLE[k])
    ax.plot(path[-1, k, 0], path[-1, k, 1], "o", ms=4.5, color=CYCLE[k])
ax.plot(centres[:, 0], centres[:, 1], "x", color=PALETTE["ink"], ms=6, mew=1.4)
ax.set_aspect("equal")
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-5.5, 6.5)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title(r"Paths of the variational means ($B=10$)")

ax = axes[1]
cx, cy = np.array(cavi_x[1:13]) / N, best - np.array(cavi_y[1:13])
keep = cy > 1e-9
ax.plot(cx[keep], cy[keep], color=PALETTE["ink"], marker="s", ms=3.5, lw=1.4, label="batch coordinate ascent")
for k, B in enumerate((10, 100)):
    xs, ys, _ = runs[B]
    ax.plot(xs[1:] / N, np.maximum(best - ys[1:], 1e-9), color=CYCLE[k], lw=1.4, label=rf"SVI, minibatch $B={B}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(4e-4, 13)
ax.set_ylim(1e-7, 10)
ax.set_xlabel("passes through the data")
ax.set_ylabel(r"$(\mathcal{L}^\star - \mathcal{L})\,/\,N$")
ax.set_title("Suboptimality against data processed")
ax.legend(loc="lower left")
save(fig, __file__, "svi-mixture")
