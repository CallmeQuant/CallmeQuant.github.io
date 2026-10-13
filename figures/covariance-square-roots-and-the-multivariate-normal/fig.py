"""Three square roots of the same covariance matrix applied to the same standard normal draws
(note: square roots of a covariance matrix)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1729)
Sigma = np.array([[2.0, 1.2], [1.2, 1.0]])
lam, U = np.linalg.eigh(Sigma)
lam, U = lam[::-1], U[:, ::-1]
if U[0, 0] < 0: U[:, 0] *= -1
if U[1, 1] < 0: U[:, 1] *= -1
factors = [
    (r"(a) $Z \sim N(0, I_2)$", np.eye(2)),
    (r"(b) Cholesky: $LZ$", np.linalg.cholesky(Sigma)),
    (r"(c) symmetric root: $\Sigma^{1/2}Z$", U @ np.diag(np.sqrt(lam)) @ U.T),
    (r"(d) eigen factor: $U\Lambda^{1/2}Z$", U @ np.diag(np.sqrt(lam))),
]
n = 400
Z = rng.standard_normal((2, n))
ang = np.arctan2(Z[1], Z[0])
t = np.linspace(0, 2 * np.pi, 300)
circ = np.vstack([np.cos(t), np.sin(t)])
e1 = np.array([1.0, 0.0])

fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), sharex=True, sharey=True)
for ax, (title, A) in zip(axes.ravel(), factors):
    X = A @ Z
    ax.scatter(X[0], X[1], c=ang, cmap="twilight", s=8, lw=0, alpha=0.9)
    for r in (1, 2):
        E = A @ (r * circ)
        ax.plot(E[0], E[1], color=PALETTE["ink"], lw=0.8)
    for e, col in ((np.array([1.0, 0.0]), PALETTE["red"]), (np.array([0.0, 1.0]), PALETTE["green"])):
        ax.annotate("", xy=2 * (A @ e), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=12))
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.8, 3.8)
for ax in axes[1]: ax.set_xlabel(r"$x_1$")
for ax in axes[:, 0]: ax.set_ylabel(r"$x_2$")
save(fig, __file__, "three-square-roots", fmt="png")
print(factors[1][1], factors[2][1], factors[3][1])
