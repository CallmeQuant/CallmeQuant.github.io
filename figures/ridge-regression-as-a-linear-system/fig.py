"""Ridge regression with p > n: coefficient paths approaching the minimum-norm least-squares
solution, and the leave-one-out shortcut checked against brute-force refitting
(note: ridge regression as a linear system)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1)
n, p = 30, 60
rho = 0.6
Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))   # AR(1) correlation
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
beta = np.zeros(p); beta[:5] = [2.0, -1.5, 1.0, 0.8, -0.6]
y = X @ beta + 2.0 * rng.standard_normal(n)

U, d, Vt = np.linalg.svd(X, full_matrices=False)
Uty = U.T @ y
lams = np.logspace(-3, 4, 200)
paths = np.array([Vt.T @ (d / (d**2 + lam) * Uty) for lam in lams])
bmin = Vt.T @ (Uty / d)                       # minimum-norm least-squares solution X^+ y

def loo_shortcut(lam):
    H = U @ np.diag(d**2 / (d**2 + lam)) @ U.T
    r = y - H @ y
    return np.mean((r / (1 - np.diag(H)))**2)

def loo_brute(lam):
    e = []
    for i in range(n):
        m = np.arange(n) != i
        b = np.linalg.solve(X[m].T @ X[m] + lam * np.eye(p), X[m].T @ y[m])
        e.append(y[i] - X[i] @ b)
    return np.mean(np.square(e))

def gcv(lam):
    H = U @ np.diag(d**2 / (d**2 + lam)) @ U.T
    r = y - H @ y
    return np.mean(r**2) / (1 - np.trace(H) / n)**2

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
for j in range(5, p):
    ax.plot(lams, paths[:, j], color=PALETTE["light"], lw=0.8)
for j in range(5):
    ax.plot(lams, paths[:, j], color=CYCLE[j], lw=1.6)
    ax.plot(lams[0] * 0.55, bmin[j], marker="<", color=CYCLE[j], ms=6, clip_on=False)
ax.set_xscale("log")
ax.set_xlim(lams[0] * 0.45, lams[-1])
ax.axhline(0, color=PALETTE["ink"], lw=0.6)
ax.set_xlabel(r"$\lambda$ (log scale)")
ax.set_ylabel(r"$\hat\beta_{\lambda,j}$")
ax.set_title(r"(a) Paths; triangles mark $(X^{+}y)_j$", loc="left")

ax = axes[1]
lg = np.logspace(-2, 3, 80)
ax.plot(lg, [loo_shortcut(l) for l in lg], color=PALETTE["blue"], label="LOO (shortcut)")
lb = np.logspace(-2, 3, 11)
ax.plot(lb, [loo_brute(l) for l in lb], "o", color=PALETTE["red"], ms=5, mfc="white", mew=1.4,
        label=r"LOO ($n$ refits)")
ax.plot(lg, [gcv(l) for l in lg], color=PALETTE["gold"], ls="--", label="GCV")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (log scale)")
ax.set_ylabel("prediction error estimate")
ax.set_title("(b) Leave-one-out error", loc="left")
ax.legend(loc="lower left")
save(fig, __file__, "paths-and-loo")
