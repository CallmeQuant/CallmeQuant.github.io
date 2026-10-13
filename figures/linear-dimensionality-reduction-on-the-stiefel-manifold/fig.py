"""Linear dimensionality reduction as optimization over orthonormal frames: a trace-ratio
(Fisher-type) objective maximized by Riemannian gradient ascent on the Stiefel manifold, compared
with PCA and with orthonormalized generalized eigenvectors
(note: linear dimensionality reduction on the Stiefel manifold)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(0)
d, r, nper = 6, 2, 100
A = rng.standard_normal((d, d)) @ np.diag([3.0, 2.5, 2.0, 1.0, 0.6, 0.4])   # common within-class factor
means = rng.standard_normal((3, d))
X = np.vstack([rng.standard_normal((nper, d)) @ A.T + means[c] for c in range(3)])
lab = np.repeat(np.arange(3), nper)
mu = X.mean(0)
Sw = sum((X[lab == c] - X[lab == c].mean(0)).T @ (X[lab == c] - X[lab == c].mean(0)) for c in range(3)) / len(X)
Sb = sum((lab == c).mean() * np.outer(X[lab == c].mean(0) - mu, X[lab == c].mean(0) - mu) for c in range(3))
St = (X - mu).T @ (X - mu) / len(X)


def f(M): return np.trace(M.T @ Sb @ M) / np.trace(M.T @ Sw @ M)
def qf(B):
    Q, R = np.linalg.qr(B); return Q * np.sign(np.diag(R))
def tangent(M, G):
    S = M.T @ G; return G - M @ (S + S.T) / 2
def egrad(M):
    a, b = np.trace(M.T @ Sb @ M), np.trace(M.T @ Sw @ M)
    return 2 * Sb @ M / b - 2 * a / b**2 * Sw @ M


def ascend(M, iters=200):
    vals = [f(M)]
    for _ in range(iters):
        Z = tangent(M, egrad(M)); t = 1.0
        while f(qf(M + t * Z)) < f(M) + 1e-4 * t * np.sum(Z * Z) and t > 1e-10:
            t /= 2                                     # Armijo backtracking along the retraction
        M = qf(M + t * Z); vals.append(f(M))
    return M, np.array(vals)


M_pca = eigh(St)[1][:, ::-1][:, :r]
M_gev = qf(eigh(Sb, Sw)[1][:, ::-1][:, :r])
runs = [ascend(qf(rng.standard_normal((d, r))), 600) for _ in range(5)]
M_opt = max(runs, key=lambda o: o[1][-1])[0]
lo, hi = 0.0, 50.0                                    # global maximum: root of rho -> sum of top-r
for _ in range(100):                                  # eigenvalues of Sb - rho * Sw
    rho = (lo + hi) / 2
    lo, hi = (rho, hi) if np.linalg.eigvalsh(Sb - rho * Sw)[-r:].sum() > 0 else (lo, rho)
rho_star = (lo + hi) / 2
print(f(M_pca), f(M_gev), [o[1][-1] for o in runs], rho_star)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
panels = [("(a) PCA", M_pca), ("(b) ratio-trace optimum (generalized eigenvectors)", M_gev),
          ("(c) trace-ratio optimum (Riemannian ascent)", M_opt)]
for ax, (title, M) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], panels):
    Y = (X - mu) @ M
    for c in range(3):
        ax.scatter(Y[lab == c, 0], Y[lab == c, 1], s=7, color=CYCLE[c], alpha=0.7, lw=0)
    ax.set_title(title + f"\n$f(M) = {f(M):.2f}$", loc="left", fontsize=10)
    ax.set_xlabel(r"$y_1$"); ax.set_ylabel(r"$y_2$")
    ax.set_aspect("equal", adjustable="datalim")
ax = axes[1, 1]
for i, (_, v) in enumerate(runs):
    ax.plot(v, color=PALETTE["blue"], lw=1.1, alpha=0.8, label="Riemannian ascent, 5 starts" if i == 0 else None)
ax.axhline(rho_star, color=PALETTE["ink"], ls="-", lw=0.8, label=r"global maximum $\rho_\star$")
ax.axhline(f(M_gev), color=PALETTE["gold"], ls="--", lw=1.2, label="ratio-trace optimum")
ax.axhline(f(M_pca), color=PALETTE["grey"], ls=":", lw=1.2, label="PCA")
ax.set_xscale("symlog", linthresh=1)
ax.set_xlim(0, 600)
ax.set_ylim(-0.2, 5.9)
ax.set_xlabel(r"iteration $k$ (log scale after 1)")
ax.set_ylabel(r"$f(M_k)$")
ax.set_title("(d) Objective along the iterations", loc="left", fontsize=10)
ax.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.0, 0.58))
save(fig, __file__, "trace-ratio-projections")
