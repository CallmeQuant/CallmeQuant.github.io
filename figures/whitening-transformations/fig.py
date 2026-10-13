"""Whitening a uniform parallelogram: ZCA, PCA and Cholesky whitening map it to differently
rotated squares (note: whitening transformations)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(404)
n = 1500
S = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(2, n))      # identity covariance
M = np.array([[1.2, 0.7], [0.25, 0.6]])
X = M @ S                                                  # population covariance M M^T
Sigma = M @ M.T
lam, U = np.linalg.eigh(Sigma); lam, U = lam[::-1], U[:, ::-1]
U = U * np.sign(U[0])                                      # first coordinates positive
W = {
    "ZCA": U @ np.diag(lam**-0.5) @ U.T,
    "PCA": np.diag(lam**-0.5) @ U.T,
    "Cholesky": np.linalg.inv(np.linalg.cholesky(Sigma)),
}
def dist(Wm):   # E || W Y - Y ||^2 for Y with covariance Sigma
    return 2 + np.trace(Sigma) - 2 * np.trace(Wm @ Sigma)

square = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]]) * np.sqrt(3)
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.3), sharex=True, sharey=True)
panels = [("(a) data $X$", np.eye(2))] + [
    (rf"({c}) {k}: $\mathbb{{E}}\Vert Z-X\Vert^2={dist(Wm):.2f}$", Wm) for c, (k, Wm) in zip("bcd", W.items())]
for ax, (title, Wm) in zip(axes.ravel(), panels):
    Y = Wm @ X
    ax.scatter(Y[0], Y[1], c=S[0], cmap="viridis", s=4, lw=0, alpha=0.8)
    B = Wm @ M @ square
    ax.plot(B[0], B[1], color=PALETTE["ink"], lw=0.9)
    corner = Wm @ M @ np.array([np.sqrt(3), np.sqrt(3)])
    ax.plot(*corner, "o", color=PALETTE["red"], ms=5)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
for ax in axes[1]: ax.set_xlabel("first coordinate")
for ax in axes[:, 0]: ax.set_ylabel("second coordinate")
save(fig, __file__, "whitened-parallelogram", fmt="png")
print({k: round(dist(v), 3) for k, v in W.items()})
