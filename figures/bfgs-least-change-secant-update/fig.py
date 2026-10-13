"""BFGS on the Rosenbrock function versus gradient descent and Newton's method, and the
hereditary property of BFGS on a quadratic (note: BFGS as a least-change secant method)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()


def f(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
def grad(x): return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
def hess(x): return np.array([[2 - 400 * x[1] + 1200 * x[0]**2, -400 * x[0]], [-400 * x[0], 200.0]])


def wolfe(x, p, g, c1=1e-4, c2=0.9, amax=50.0):
    """Bracketing/bisection line search for the weak Wolfe conditions, starting at a = 1."""
    lo, hi, a = 0.0, np.inf, 1.0
    f0, d0 = f(x), g @ p
    for _ in range(60):
        if f(x + a * p) > f0 + c1 * a * d0:
            hi = a
        elif grad(x + a * p) @ p < c2 * d0:
            lo = a
        else:
            return a
        a = (lo + hi) / 2 if np.isfinite(hi) else min(2 * a, amax)
    return a


def backtrack(x, p, g, c1=1e-4):
    a = 1.0
    while f(x + a * p) > f(x) + c1 * a * (g @ p):
        a *= 0.5
    return a


def run(method, x0, iters, tol=1e-10):
    x = x0.copy(); H = np.eye(2); xs = [x.copy()]; gs = [np.linalg.norm(grad(x))]
    for _ in range(iters):
        g = grad(x)
        if np.linalg.norm(g) < tol: break
        if method == "gd":
            p = -g; a = 1e-3                 # fixed step below 2 / (largest Hessian eigenvalue on the path)
        elif method == "newton":
            p = -np.linalg.solve(hess(x), g); a = backtrack(x, p, g)
        else:
            p = -H @ g; a = wolfe(x, p, g)
        xn = x + a * p
        if method == "bfgs":
            s, y = xn - x, grad(xn) - g
            rho = 1 / (y @ s)
            if len(xs) == 1:           # standard initial scaling H_0 = (y's / y'y) I
                H = (y @ s) / (y @ y) * np.eye(2)
            V = np.eye(2) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        x = xn; xs.append(x.copy()); gs.append(np.linalg.norm(grad(x)))
    return np.array(xs), np.array(gs)


x0 = np.array([-1.2, 1.0])
runs = {"gradient descent": run("gd", x0, 3000), "BFGS": run("bfgs", x0, 200), "Newton": run("newton", x0, 200)}
cols = {"gradient descent": PALETTE["red"], "BFGS": PALETTE["blue"], "Newton": PALETTE["green"]}

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
g1, g2 = np.meshgrid(np.linspace(-1.6, 1.5, 400), np.linspace(-0.6, 1.6, 400))
F = (1 - g1)**2 + 100 * (g2 - g1**2)**2
ax.contour(g1, g2, F, levels=np.geomspace(0.05, 800, 14), colors=PALETTE["grey"], linewidths=0.6)
for name, (xs, _) in runs.items():
    ax.plot(xs[:, 0], xs[:, 1], "-", color=cols[name], lw=1.3 if name != "gradient descent" else 0.9,
            marker="o" if name != "gradient descent" else None, ms=2.6, label=f"{name} ({len(xs) - 1} it.)")
ax.plot(1, 1, "*", color=PALETTE["ink"], ms=11, zorder=5)
ax.plot(*x0, "s", color=PALETTE["ink"], ms=5)
ax.set_xlim(-1.6, 1.5); ax.set_ylim(-0.6, 1.6)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title("(a) Rosenbrock function, start $(-1.2, 1)$", loc="left")
gd_end = runs["gradient descent"][0][-1]
ax.plot(*gd_end, "o", color=PALETTE["red"], ms=4.5, zorder=6)
leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="none")
leg.get_frame().set_facecolor("white")

ax = axes[1]
for name, (_, gs) in runs.items():
    ax.semilogy(np.arange(len(gs)), gs, color=cols[name], label=name)
gd_final = runs["gradient descent"][1][-1]
ax.text(58, 4e-3, "gradient descent:\n" + rf"${gd_final:.2f}$ after 3000 it.", ha="right", va="top",
        color=PALETTE["red"], fontsize=9)
ax.set_xlim(0, 60)
ax.set_ylim(1e-11, 1e4)
ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"$\Vert\nabla f(x_k)\Vert$")
ax.set_title("(b) Gradient norm", loc="left")
save(fig, __file__, "rosenbrock")
print({k: len(v[0]) - 1 for k, v in runs.items()})

# ---- Figure 2: hereditary property on a quadratic -------------------------------------------
rng = np.random.default_rng(8)
n = 8
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
A = (Q * np.logspace(0.3, 2.3, n)) @ Q.T
b = rng.standard_normal(n)
x = np.zeros(n); H = np.eye(n)
eigs = [np.sort(np.linalg.eigvals(H @ A).real)]
for k in range(n):
    g = A @ x - b
    p = -H @ g
    a = -(g @ p) / (p @ A @ p)        # exact line search
    s = a * p; y = A @ s
    rho = 1 / (y @ s); V = np.eye(n) - rho * np.outer(s, y)
    H = V @ H @ V.T + rho * np.outer(s, s)
    x = x + s
    eigs.append(np.sort(np.linalg.eigvals(H @ A).real))
eigs = np.array(eigs)
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for k in range(n + 1):
    e = eigs[k]
    one = np.isclose(e, 1.0, atol=1e-6)
    ax.scatter(np.full((~one).sum(), k), e[~one], color=PALETTE["grey"], s=18, zorder=3)
    ax.scatter(np.full(one.sum(), k), e[one], color=PALETTE["blue"], s=22, zorder=4)
    ax.text(k, 400, f"{one.sum()}", ha="center", va="bottom", fontsize=9, color=PALETTE["blue"])
ax.axhline(1, color=PALETTE["blue"], lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_ylim(0.5, 900)
ax.set_xticks(range(n + 1))
ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"eigenvalues of $H_k A$")
ax.set_title(r"Quadratic, $n = 8$, exact line search; top: number of eigenvalues equal to $1$", loc="left", fontsize=10.5)
save(fig, __file__, "hereditary-eigenvalues")
