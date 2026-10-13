"""Newton's method versus Fisher scoring for a gamma GLM with identity link (note: Newton and Fisher
scoring for GLMs). Left: log-likelihood contours, the region where the observed information is not
positive definite, and iterates from two starting points. Right: error against iteration near the MLE."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1972)
n, NU = 60, 2.0                                  # gamma shape (known), so phi = 1 / NU
x = rng.uniform(0, 1, n)
X = np.column_stack([np.ones(n), x])
y = rng.gamma(NU, (1 + 2 * x) / NU)              # E y = 1 + 2x


def loglik(b):
    m = X @ b
    return -np.inf if np.any(m <= 0) else NU * np.sum(-y / m - np.log(m))


def score(b):
    m = X @ b
    return NU * X.T @ ((y - m) / m**2)


def obs_info(b):                                 # minus the Hessian
    m = X @ b
    return NU * (X * ((2 * y - m) / m**3)[:, None]).T @ X


def exp_info(b):
    m = X @ b
    return NU * (X / m[:, None] ** 2).T @ X


def iterate(b, method, iters):
    path = [np.array(b, float)]
    for _ in range(iters):
        b = path[-1]
        M = obs_info(b) if method == "newton" else exp_info(b)
        step = np.linalg.solve(M, score(b))
        s = 1.0
        while loglik(b + s * step) == -np.inf:   # halve only to stay inside the parameter space
            s /= 2
        path.append(b + s * step)
    return np.array(path)


bhat = iterate([1.0, 2.0], "fisher", 60)[-1]

fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
b0, b1 = np.meshgrid(np.linspace(0.02, 8, 300), np.linspace(-8, 8, 300))
L = np.array([[loglik(np.array([u, v])) for u, v in zip(r0, r1)] for r0, r1 in zip(b0, b1)])
notnd = np.array([[np.linalg.eigvalsh(obs_info(np.array([u, v]))).min() <= 0 if np.isfinite(loglik(np.array([u, v]))) else False
                   for u, v in zip(r0, r1)] for r0, r1 in zip(b0, b1)])
ax.contourf(b0, b1, np.where(notnd, 1.0, np.nan), levels=[0.5, 1.5], colors=[PALETTE["gold"]], alpha=0.22)
lmax = loglik(bhat)
ax.contour(b0, b1, lmax - L, levels=[2, 10, 30, 60, 100, 160, 240], colors=PALETTE["grey"], linewidths=0.6)
ax.fill_between([0, 8], [0, -8], [-8, -8], color=PALETTE["light"], alpha=0.8, lw=0)
ax.text(5.2, -7.2, r"$\beta_0 + \beta_1 \leq 0$", fontsize=9, color=PALETTE["ink"])
ax.text(4.75, 4.3, "observed information\nnot positive definite", fontsize=8.6, color="#8a6418", ha="left", va="top")
ax.text(6.9, 7.55, "Newton diverges", fontsize=8.6, color=PALETTE["red"], ha="right", va="center")
for start, mk in [((6.0, 6.0), "o"), ((4.0, 0.5), "s")]:
    for method, col in [("newton", PALETTE["red"]), ("fisher", PALETTE["blue"])]:
        P = iterate(start, method, 12)
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.2, marker=mk, ms=3.5, clip_on=True,
                label=None)
    ax.plot(*start, mk, color=PALETTE["ink"], ms=6)
ax.plot(*bhat, "*", color=PALETTE["ink"], ms=10)
ax.plot([], [], color=PALETTE["red"], label="Newton")
ax.plot([], [], color=PALETTE["blue"], label="Fisher scoring")
ax.set_xlim(0, 8); ax.set_ylim(-8, 8)
ax.set_xlabel(r"$\beta_0$"); ax.set_ylabel(r"$\beta_1$")
ax.set_title(r"Gamma GLM, identity link, $n = 60$", loc="left")
ax.legend(loc="lower left", fontsize=8.8, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
ax.grid(False)

ax = axes[1]
start = np.array([2.0, 0.3])
for method, col, mk, lab in [("newton", PALETTE["red"], "o", "Newton"), ("fisher", PALETTE["blue"], "s", "Fisher scoring")]:
    P = iterate(start, method, 10)
    err = np.linalg.norm(P - bhat, axis=1)
    keep = err > 1e-14
    ax.semilogy(np.arange(len(err))[keep], err[keep], "-" + mk, color=col, mfc="white", mew=1.2, ms=4.5, label=lab)
J, I = obs_info(bhat), exp_info(bhat)
rho = np.max(np.abs(np.linalg.eigvals(np.eye(2) - np.linalg.solve(I, J))))
t = np.arange(0, 11)
ax.semilogy(t, 0.135 * rho ** (t - 1.0), ls=":", color=PALETTE["grey"], lw=1.2,
            label=rf"$\rho^t$, $\rho = {rho:.3f}$")
ax.set_ylim(1e-12, 10)
ax.set_xlabel("iteration $t$")
ax.set_ylabel(r"$\Vert \beta^{(t)} - \hat\beta \Vert$")
ax.set_title(r"Local convergence from $(2, 0.3)$", loc="left")
ax.legend(loc="lower left", fontsize=8.8)
save(fig, __file__, "newton-vs-scoring")
print("bhat", bhat, "rho", rho)
