"""Power iteration on symmetric 200 x 200 matrices: angle error for several eigenvalue ratios,
and the squared rate of the Rayleigh quotient (note: power iteration and the eigengap)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(11)
n, K = 200, 150
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
x0 = rng.standard_normal(n)
FLOOR = 1e-15


def make(r):
    lam = np.concatenate([[1.0, r], rng.uniform(-0.5 * r, 0.5 * r, n - 2)])
    return (V * lam) @ V.T, lam


def run(A, lam, K):
    v1 = V[:, 0]
    x = x0 / np.linalg.norm(x0)
    tan, rq = [], []
    for k in range(K + 1):
        c = v1 @ x
        tan.append(np.linalg.norm(x - c * v1) / abs(c))       # stable tan(theta_k)
        rq.append(abs(x @ A @ x - lam[0]))
        y = A @ x
        x = y / np.linalg.norm(y)
    return np.array(tan), np.array(rq)


def cut(v):
    v = v.copy()
    idx = np.argmax(v < FLOOR) if np.any(v < FLOOR) else None
    if idx is not None:
        v[idx + 1:] = np.nan
    return v


ratios = [0.5, 0.8, 0.9, 0.95, 0.99]
k = np.arange(K + 1)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
ax = axes[0]
for i, r in enumerate(ratios):
    A, lam = make(r)
    tan = cut(run(A, lam, K)[0])
    ax.semilogy(k, tan, color=CYCLE[i])
    last = np.where(np.isfinite(tan))[0][-1]
    ax.text(k[last] + 3, max(tan[last], 3e-16), f"{r}", color=CYCLE[i], va="center", fontsize=9)
ax.set_xlim(0, 172)
ax.set_ylim(1e-16, 1e3)
ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"$\tan\theta_k$")
ax.set_title(r"(a) Angle to $v_1$ for $\lambda_2/\lambda_1$ as labelled", loc="left", fontsize=10.5)

ax = axes[1]
r = 0.9
A, lam = make(r)
tan, rq = run(A, lam, K)
sin = cut(tan / np.sqrt(1 + tan**2)); rq = cut(rq)
ax.semilogy(k, sin, color=PALETTE["blue"], label=r"$\sin\theta_k$")
ax.semilogy(k, rq, color=PALETTE["red"], label=r"$|\rho(x_k)-\lambda_1|$")
ax.set_ylim(1e-16, 3)
ax.set_xlabel(r"iteration $k$")
ax.set_title(r"(b) $\lambda_2/\lambda_1 = 0.9$: eigenvalue error is squared", loc="left", fontsize=10.5)
ax.legend(loc="lower left")
save(fig, __file__, "power-iteration-rates")
