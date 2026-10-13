"""EM for a three-component bivariate Gaussian mixture (note: em-application1).
Figure 1 (gmm-data): complete data coloured by component versus the observed data.
Figure 2 (em-iterations): fitted components at several EM iterations and the log-likelihood trace."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.special import logsumexp
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(91)
w_true = np.array([0.4, 0.3, 0.3])
mu_true = np.array([[-3.0, 0.0], [2.0, 2.5], [2.5, -2.0]])
cov_true = np.array([[[1.2, 0.6], [0.6, 1.0]],
                     [[0.8, -0.3], [-0.3, 0.6]],
                     [[1.5, 0.0], [0.0, 0.4]]])
n = 450
z = rng.choice(3, size=n, p=w_true)
X = np.array([rng.multivariate_normal(mu_true[k], cov_true[k]) for k in z])
COLS = [PALETTE["blue"], PALETTE["red"], PALETTE["green"]]
RGB = np.array([to_rgb(c) for c in COLS])  # point colour = responsibility-weighted mix


def log_gauss(X, mu, S):
    L = np.linalg.cholesky(S)
    sol = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (sol**2).sum(0) - np.log(np.diag(L)).sum() - X.shape[1] / 2 * np.log(2 * np.pi)


def e_step(X, w, mu, S):
    logp = np.column_stack([np.log(w[k]) + log_gauss(X, mu[k], S[k]) for k in range(len(w))])
    ll = logsumexp(logp, axis=1)
    return np.exp(logp - ll[:, None]), ll.sum()


def m_step(X, g):
    Nk = g.sum(0)
    mu = (g.T @ X) / Nk[:, None]
    S = np.array([((g[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] for k in range(g.shape[1])])
    return Nk / len(X), mu, S


def ellipse(mu, S, level=5.991):
    vals, vecs = np.linalg.eigh(S)
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.column_stack([np.cos(th), np.sin(th)]) * np.sqrt(level * vals)
    return mu + circ @ vecs.T


# Figure 1: complete versus observed data
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
for k in range(3):
    axes[0].scatter(*X[z == k].T, s=9, color=COLS[k], alpha=0.75, lw=0, label=fr"$z_i = {k + 1}$")
axes[0].set_title(r"(a) complete data $(x_i, z_i)$", loc="left")
axes[0].legend(loc="upper left", handletextpad=0.2, markerscale=1.6)
axes[1].scatter(*X.T, s=9, color=PALETTE["grey"], alpha=0.75, lw=0)
axes[1].set_title(r"(b) observed data $x_i$", loc="left")
for ax in axes:
    ax.set_xlabel(r"$x_{i1}$")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$x_{i2}$")
save(fig, __file__, "gmm-data")

# EM run from a poor start: two initial means in the same cluster, pooled covariance
init_idx = np.array([np.flatnonzero(z == 0)[0], np.flatnonzero(z == 0)[1], np.flatnonzero(z == 2)[0]])
w, mu, S = np.full(3, 1 / 3), X[init_idx].copy(), np.array([np.cov(X.T)] * 3)
snaps, lls = {0: (w, mu, S)}, []
for it in range(1, 201):
    g, ll = e_step(X, w, mu, S)
    lls.append(ll)
    w, mu, S = m_step(X, g)
    snaps[it] = (w, mu, S)
    if it > 1 and lls[-1] - lls[-2] < 1e-8:
        break
T = it
show = [0, 1, 4, 10, T]

fig = plt.figure(figsize=(8.0, 5.4))
gs = fig.add_gridspec(2, 3)
for j, t in enumerate(show):
    ax = fig.add_subplot(gs[j // 3, j % 3])
    w, mu, S = snaps[t]
    g, _ = e_step(X, w, mu, S)
    ax.scatter(*X.T, s=5, c=g @ RGB,
               alpha=0.6, lw=0)
    for k in range(3):
        e = ellipse(mu[k], S[k])
        ax.plot(*e.T, color=PALETTE["ink"], lw=1.1)
        ax.scatter(*mu[k], marker="x", s=30, color=PALETTE["ink"], lw=1.4, zorder=3)
    ax.set_title(f"iteration {t}", loc="left")
    ax.set_xlim(-7, 7); ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
ax = fig.add_subplot(gs[1, 2])
ax.plot(np.arange(0, len(lls)), lls, color=PALETTE["blue"], marker="o", ms=2.5, lw=1.2)
ax.set_xlabel("iteration $t$")
ax.set_title(r"log-likelihood $\ell(\Theta_t)$", loc="left")
ax.tick_params(labelsize=8)
save(fig, __file__, "em-iterations")
print("iterations", T, "final weights", snaps[T][0].round(3), "means", snaps[T][1].round(2))
