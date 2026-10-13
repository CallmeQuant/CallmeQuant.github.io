"""Cholesky factors of the covariance and of the precision matrix of a stationary AR(1) vector,
and draws obtained from the precision factor by back-substitution (note: sampling the multivariate normal)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_banded
from style import use_style, PALETTE, CYCLE, save

use_style()
phi = 0.9


def ar1_cov(d):
    i = np.arange(d)
    return phi ** np.abs(i[:, None] - i[None, :]) / (1 - phi**2)


def ar1_prec(d):
    Q = np.diag(np.r_[1.0, np.full(d - 2, 1 + phi**2), 1.0])
    return Q - phi * (np.eye(d, k=1) + np.eye(d, k=-1))


d_small = 30
L_S = np.linalg.cholesky(ar1_cov(d_small))
L_Q = np.linalg.cholesky(ar1_prec(d_small))
cmap = LinearSegmentedColormap.from_list("blog", ["#ffffff", PALETTE["blue"], PALETTE["ink"]])

fig = plt.figure(figsize=(8.6, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.9])
for k, (M, title) in enumerate([(L_S, r"(a) $L_\Sigma$, $\Sigma = L_\Sigma L_\Sigma^\top$"),
                                (L_Q, r"(b) $L_Q$, $Q = \Sigma^{-1} = L_Q L_Q^\top$")]):
    ax = fig.add_subplot(gs[0, k])
    A = np.ma.masked_where(np.abs(M) < 1e-12, np.abs(M))
    ax.imshow(A, cmap=cmap, vmin=0, vmax=np.abs(M).max(), interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xticks([0, d_small - 1]); ax.set_xticklabels(["1", str(d_small)])
    ax.set_yticks([0, d_small - 1]); ax.set_yticklabels(["1", str(d_small)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(PALETTE["grey"]); s.set_linewidth(0.6)
    nnz = int(np.sum(np.abs(M) > 1e-12))
    ax.text(0.97, 0.97, f"{nnz} nonzeros", transform=ax.transAxes, ha="right", va="top", fontsize=9)

# (c) draws via back-substitution with the bidiagonal factor of Q
d = 300
rng = np.random.default_rng(7)
Qfull = ar1_prec(d)
Lq = np.linalg.cholesky(Qfull)
# banded storage of the upper-triangular L_Q^T: row 0 = superdiagonal, row 1 = diagonal
ab = np.zeros((2, d))
ab[0, 1:] = np.diag(Lq, -1)
ab[1, :] = np.diag(Lq)
ax = fig.add_subplot(gs[0, 2])
sd = 1 / np.sqrt(1 - phi**2)
t = np.arange(1, d + 1)
ax.fill_between(t, -2 * sd, 2 * sd, color=PALETTE["light"], lw=0)
for j in range(3):
    z = rng.standard_normal(d)
    y = solve_banded((0, 1), ab, z)          # solves L_Q^T y = z in O(d) operations
    ax.plot(t, y, color=CYCLE[j], lw=1.0)
ax.set_xlim(1, d)
ax.set_ylim(-8.5, 8.5)
ax.set_xlabel(r"coordinate $i$")
ax.set_title(r"(c) draws $y = L_Q^{-\top} z$, $d = 300$", loc="left", fontsize=10.5)
save(fig, __file__, "precision-cholesky")

# check: covariance of y = L_Q^{-T} z equals Sigma; L_Q^{-1} z does not
Linv = np.linalg.inv(L_Q)
print(np.allclose(Linv.T @ Linv, ar1_cov(d_small)), np.allclose(Linv @ Linv.T, ar1_cov(d_small)))
