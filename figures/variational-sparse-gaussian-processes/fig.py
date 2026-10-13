"""Titsias's collapsed variational bound for sparse Gaussian process regression: the bound against the
exact log marginal likelihood for nested inducing sets, and the effect of optimizing the inducing inputs
(note: a variational lower bound for sparse Gaussian processes)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2009)
ELL, SF, SN = 0.6, 1.0, 0.25


def kern(a, b):
    return SF ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ELL ** 2)


def f_true(x):
    return 0.8 * np.sin(2 * x) + 0.4 * np.cos(5 * x)


n = 250
X = np.sort(rng.uniform(0, 8, n))
y = f_true(X) + SN * rng.standard_normal(n)


def log_mvn_lowrank(Z):
    """log N(y | 0, Q_ff + SN^2 I) and tr(K_ff - Q_ff), in O(n m^2)."""
    m = Z.size
    Kuu = kern(Z, Z) + 1e-8 * np.eye(m)
    Kuf = kern(Z, X)
    Lu = np.linalg.cholesky(Kuu)
    V = np.linalg.solve(Lu, Kuf)                              # m x n, Q_ff = V^T V
    B = np.eye(m) + V @ V.T / SN ** 2
    LB = np.linalg.cholesky(B)
    c = np.linalg.solve(LB, V @ y) / SN ** 2
    quad = y @ y / SN ** 2 - c @ c                            # y^T (Q + s^2 I)^{-1} y (Woodbury)
    logdet = 2 * np.sum(np.log(np.diag(LB))) + n * np.log(SN ** 2)   # determinant lemma
    ll = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    trace = n * SF ** 2 - np.sum(V ** 2)
    return ll, trace


def bound(Z):
    ll, tr = log_mvn_lowrank(Z)
    return ll - tr / (2 * SN ** 2)


c_ex = cho_factor(kern(X, X) + SN ** 2 * np.eye(n), lower=True)
log_ev = -0.5 * y @ cho_solve(c_ex, y) - np.sum(np.log(np.diag(c_ex[0]))) - 0.5 * n * np.log(2 * np.pi)
print("exact log p(y) =", log_ev)

# ---------------- Figure 1: nested inducing sets drawn from the training inputs -----------------
perm = rng.permutation(n)
ms = np.arange(1, 61)
L_vals, dtc_vals, tr_vals = [], [], []
for m in ms:
    Z = X[perm[:m]]
    ll, tr = log_mvn_lowrank(Z)
    L_vals.append(ll - tr / (2 * SN ** 2))
    dtc_vals.append(ll)
    tr_vals.append(tr)
L_vals, dtc_vals, tr_vals = map(np.array, (L_vals, dtc_vals, tr_vals))

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axes[0]
ax.axhline(log_ev, color=PALETTE["ink"], lw=1.2, ls="--", label=r"exact $\log p(y)$")
ax.plot(ms, dtc_vals, color=PALETTE["gold"], lw=1.4, label=r"$\log\mathcal{N}(y\mid 0, Q_{ff}+\sigma^2 I)$")
ax.plot(ms, L_vals, color=PALETTE["blue"], lw=1.8, label=r"bound $\mathcal{L}^\ast(Z)$")
ax.set_xlim(1, 60)
ax.set_ylim(-420, -30)
ax.set_xlabel(r"number of inducing inputs $m$")
ax.set_ylabel("nats")
ax.set_title("Bound and approximate likelihood", loc="left")
ax.legend(loc="lower right", fontsize=8.8)
ax = axes[1]
ax.semilogy(ms, log_ev - L_vals, color=PALETTE["blue"], lw=1.8, label=r"gap $\log p(y)-\mathcal{L}^\ast$")
ax.semilogy(ms, tr_vals / (2 * SN ** 2), color=PALETTE["red"], lw=1.4, ls="--",
            label=r"$\mathrm{tr}(K_{ff}-Q_{ff})/(2\sigma^2)$")
ax.set_xlim(1, 60)
ax.set_xlabel(r"number of inducing inputs $m$")
ax.set_title("Gap and trace penalty", loc="left")
ax.legend(loc="upper right", fontsize=8.8)
save(fig, __file__, "bound-vs-m")
print("gap at m=10,20,40:", (log_ev - L_vals)[[9, 19, 39]])

# ---------------- Figure 2: optimizing the inducing inputs ---------------------------------------
def predict(xq, Z):
    """Predictive mean and variance of f under the optimal q(u) (identical to DTC)."""
    m = Z.size
    Kuu = kern(Z, Z) + 1e-8 * np.eye(m)
    Kuf = kern(Z, X)
    Ksu = kern(xq, Z)
    S = cho_factor(Kuu + Kuf @ Kuf.T / SN ** 2, lower=True)
    cu = cho_factor(Kuu, lower=True)
    mean = Ksu @ cho_solve(S, Kuf @ y) / SN ** 2
    var = SF ** 2 - np.sum(Ksu * cho_solve(cu, Ksu.T).T, 1) + np.sum(Ksu * cho_solve(S, Ksu.T).T, 1)
    return mean, var


m = 16
Z0 = np.linspace(0.2, 2.0, m)
res = minimize(lambda z: -bound(z), Z0, method="L-BFGS-B", bounds=[(0.0, 8.0)] * m)
Zopt = np.sort(res.x)
print("bound initial, optimized, grid:", bound(Z0), bound(Zopt), bound(np.linspace(0, 8, m)), "Z*:", np.round(Zopt, 2))
xs = np.linspace(-0.5, 8.5, 500)
fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True, sharey=True)
for ax, Z, ttl in [(axes[0], Z0, "initial inducing inputs"), (axes[1], Zopt, "after maximizing the bound over $Z$")]:
    mu, var = predict(xs, Z)
    sd = np.sqrt(np.clip(var, 0, None))
    ax.plot(X, y, ".", color=PALETTE["grey"], ms=2.5, alpha=0.6)
    ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color=PALETTE["blue"], alpha=0.18, lw=0)
    ax.plot(xs, mu, color=PALETTE["blue"], lw=1.6)
    ax.plot(xs, f_true(xs), color=PALETTE["ink"], ls="--", lw=0.9)
    ax.plot(Z, np.full(Z.size, -2.15), "^", color=PALETTE["red"], ms=5.5, clip_on=False)
    ax.set_title(rf"{ttl}: $\mathcal{{L}}^\ast = {bound(Z):.1f}$", loc="left")
    ax.set_ylim(-2.3, 2.3)
    ax.set_ylabel(r"$f(x)$")
axes[1].set_xlim(-0.5, 8.5)
axes[1].set_xlabel(r"$x$")
save(fig, __file__, "optimized-inducing-inputs")
