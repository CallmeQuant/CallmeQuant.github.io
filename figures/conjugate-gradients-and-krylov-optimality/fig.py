"""Conjugate gradients versus steepest descent: paths on a 2-d quadratic, and A-norm error on
400-dimensional problems (note: conjugate gradients and Krylov optimality)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()


def steepest(A, b, x0, iters):
    xs = [x0.copy()]; x = x0.copy()
    for _ in range(iters):
        r = b - A @ x
        if r @ r == 0: break
        a = (r @ r) / (r @ A @ r)
        x = x + a * r; xs.append(x.copy())
    return np.array(xs)


def cg(A, b, x0, iters):
    xs = [x0.copy()]; x = x0.copy()
    r = b - A @ x; p = r.copy(); rr = r @ r
    for _ in range(iters):
        if rr == 0: break
        Ap = A @ p
        a = rr / (p @ Ap)
        x = x + a * p; r = r - a * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p; rr = rr_new
        xs.append(x.copy())
    return np.array(xs)


def krylov_optimum(A, b, x0, iters):
    """Exact-arithmetic CG iterates: minimize the A-norm error over x0 + K_k, using an orthonormal
    Krylov basis built with full reorthogonalization (Lanczos)."""
    r0 = b - A @ x0
    V = [r0 / np.linalg.norm(r0)]; xs = [x0.copy()]
    for k in range(1, iters + 1):
        Vk = np.array(V).T
        y = np.linalg.solve(Vk.T @ A @ Vk, Vk.T @ r0)
        xs.append(x0 + Vk @ y)
        w = A @ V[-1]
        for _ in range(2):
            w = w - Vk @ (Vk.T @ w)
        V.append(w / np.linalg.norm(w))
    return np.array(xs)


# ---- panel (a): 2-d quadratic -------------------------------------------------------------
th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
A2 = R @ np.diag([1.0, 10.0]) @ R.T
xstar = np.zeros(2); b2 = A2 @ xstar
x0 = R @ np.array([-2.7, 0.27])         # worst-case start for steepest descent
sd = steepest(A2, b2, x0, 25)
cgp = cg(A2, b2, x0, 2)

# ---- panel (b): n = 400 --------------------------------------------------------------------
rng = np.random.default_rng(5)
n, K = 400, 160
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
kappa = 1000.0
lam_spread = np.logspace(0, np.log10(kappa), n)
lam_clust = np.concatenate([rng.uniform(1, 2, n - 5), [200, 400, 600, 800, 1000]])
xs_true = rng.standard_normal(n)
def anorm_err(A, X):
    E = X - xs_true
    return np.sqrt(np.einsum("ij,jk,ik->i", E, A, E))
res = {}
for name, lam in [("spread", lam_spread), ("clustered", lam_clust)]:
    A = (Q * lam) @ Q.T
    b = A @ xs_true
    z = np.zeros(n)
    res[name, "cg"] = anorm_err(A, cg(A, b, z, K))
    res[name, "sd"] = anorm_err(A, steepest(A, b, z, K))
    if name == "clustered":
        res[name, "exact"] = anorm_err(A, krylov_optimum(A, b, z, 30))
e0 = res["spread", "cg"][0]
k = np.arange(K + 1)
cheb = 2 * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
sdrate = ((kappa - 1) / (kappa + 1)) ** k

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
g1, g2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2.2, 2.2, 300))
G = np.stack([g1, g2], -1)
F = 0.5 * np.einsum("...i,ij,...j->...", G, A2, G)
ax.contour(g1, g2, F, levels=np.geomspace(0.02, 30, 10), colors=PALETTE["grey"], linewidths=0.7)
ax.plot(cgp[:, 0], cgp[:, 1], "-o", color=PALETTE["blue"], ms=5, lw=3.2, alpha=0.55, label="conjugate gradients", zorder=2)
ax.plot(sd[:, 0], sd[:, 1], "-o", color=PALETTE["red"], ms=2.5, lw=1.0, label="steepest descent", zorder=3)
ax.plot(*xstar, "*", color=PALETTE["ink"], ms=10)
ax.set_aspect("equal")
ax.set_xlim(-3, 3); ax.set_ylim(-2.2, 2.2)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title(r"(a) $n=2$, $\kappa=10$", loc="left")
ax.legend(loc="lower right", fontsize=8.5, handlelength=1.4)

ax = axes[1]
ax.semilogy(k, res["spread", "sd"] / e0, color=PALETTE["red"], label="steepest descent, spread")
ax.semilogy(k, res["spread", "cg"] / e0, color=PALETTE["blue"], label="CG, spread spectrum")
ax.semilogy(k, res["clustered", "cg"] / res["clustered", "cg"][0], color=PALETTE["green"], label="CG, clustered + 5 outliers")
ax.semilogy(np.arange(31), res["clustered", "exact"] / res["clustered", "exact"][0], color=PALETTE["green"],
            ls=(0, (4, 2.5)), lw=1.4, label="same, exact arithmetic")
ax.semilogy(k, cheb, color=PALETTE["blue"], ls=":", lw=1.2, label=r"$2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$")
ax.set_ylim(1e-10, 3)
ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"$\Vert x_k-x_\star\Vert_A/\Vert x_0-x_\star\Vert_A$")
ax.set_title(r"(b) $n=400$, $\kappa=1000$", loc="left")
ax.legend(loc="lower right", fontsize=8.5)
save(fig, __file__, "cg-vs-steepest-descent")
