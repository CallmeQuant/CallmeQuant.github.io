"""Accuracy of least squares via the normal equations (Cholesky), Householder QR and the SVD
as the condition number of the design grows (note: LU, Cholesky, QR and SVD)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(31)
n, p, reps = 100, 10, 20
kappas = np.logspace(0, 9, 19)
u = np.finfo(float).eps / 2


def solve_normal(A, y):
    try:
        c = sla.cho_factor(A.T @ A, lower=True)
        return sla.cho_solve(c, A.T @ y)
    except np.linalg.LinAlgError:
        return np.full(A.shape[1], np.nan)


def solve_qr(A, y):
    Q, R = np.linalg.qr(A)                       # Householder QR (LAPACK geqrf)
    return sla.solve_triangular(R, Q.T @ y)


def solve_svd(A, y):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


methods = [("normal equations + Cholesky", solve_normal, PALETTE["red"], "o"),
           ("Householder QR", solve_qr, PALETTE["blue"], "s"),
           ("SVD", solve_svd, PALETTE["green"], "^")]
err = {m[0]: np.zeros((len(kappas), reps)) for m in methods}
for i, k in enumerate(kappas):
    for r in range(reps):
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        A = U @ np.diag(np.logspace(0, -np.log10(k), p)) @ V.T
        beta = rng.standard_normal(p); beta /= np.linalg.norm(beta)
        y = A @ beta                              # consistent system: zero residual
        for name, f, _, _ in methods:
            b = f(A, y)
            err[name][i, r] = np.linalg.norm(b - beta) if np.all(np.isfinite(b)) else np.nan

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for name, _, col, mk in methods:
    e = err[name]
    ok = np.isfinite(e).mean(1) == 1
    med = np.where(ok, np.nanmedian(e, 1), np.nan)
    ax.plot(kappas, np.minimum(med, 1.0), marker=mk, color=col, label=name, ms=5)
ax.plot(kappas, u * kappas, color=PALETTE["grey"], ls="--", lw=1)
ax.plot(kappas, u * kappas**2, color=PALETTE["grey"], ls=":", lw=1.2)
ax.text(1e6, u * 1e6 * 5, r"$u\,\kappa$", color=PALETTE["grey"], ha="right", va="bottom")
ax.text(3e5, u * 9e10 * 3, r"$u\,\kappa^2$", color=PALETTE["grey"], ha="right", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(1e-17, 3)
ax.set_xlabel(r"condition number $\kappa_2(A)=\sigma_1/\sigma_p$")
ax.set_ylabel(r"relative error $\Vert\hat\beta-\beta\Vert/\Vert\beta\Vert$")
ax.legend(loc="upper left")
save(fig, __file__, "least-squares-accuracy")
