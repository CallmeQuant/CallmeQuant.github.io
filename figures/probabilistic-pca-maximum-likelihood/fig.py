"""Probabilistic PCA: eigenvalues of the sample covariance versus those of the fitted model
covariance, and EM iterations converging to the closed-form maximum likelihood value
(note: probabilistic PCA)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1999)
n, p, q, sigma2 = 200, 20, 3, 1.0
W_true = rng.standard_normal((p, q)) * np.array([3.0, 2.0, 1.4])
X = rng.standard_normal((n, q)) @ W_true.T + np.sqrt(sigma2) * rng.standard_normal((n, p))
xbar = X.mean(0)
S = (X - xbar).T @ (X - xbar) / n
lam, U = np.linalg.eigh(S); lam, U = lam[::-1], U[:, ::-1]


def loglik(W, s2):
    C = W @ W.T + s2 * np.eye(p)
    sign, logdet = np.linalg.slogdet(C)
    return -0.5 * n * (p * np.log(2 * np.pi) + logdet + np.trace(np.linalg.solve(C, S)))


def closed_form(q):
    s2 = lam[q:].mean()
    return U[:, :q] * np.sqrt(lam[:q] - s2), s2


W_ml, s2_ml = closed_form(q)
ll_ml = loglik(W_ml, s2_ml)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axes[0]
j = np.arange(1, p + 1)
ax.plot(j, lam, "o", color=PALETTE["ink"], ms=4, label=r"eigenvalues $\lambda_j$ of $S$")
model = np.concatenate([lam[:q], np.full(p - q, s2_ml)])
ax.plot(j, model, "-", color=PALETTE["blue"], lw=1.6, drawstyle="steps-mid", label=r"eigenvalues of $\hat C$")
ax.axhline(sigma2, color=PALETTE["red"], ls=":", lw=1.2, label=r"true $\sigma^2 = 1$")
ax.set_yscale("log")
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel(r"index $j$")
ax.set_ylabel("eigenvalue (log scale)")
ax.set_title(rf"(a) $p={p}$, $q={q}$: $\hat\sigma^2 = {s2_ml:.2f}$", loc="left")
ax.legend(loc="upper right", fontsize=8.5)

ax = axes[1]
T = 3000
P_true = U[:, :q] @ U[:, :q].T
for i in range(4):
    W = rng.standard_normal((p, q)); s2 = 5.0
    gaps, sub = [], []
    for it in range(T):
        gaps.append(ll_ml - loglik(W, s2))
        sub.append(np.linalg.norm(W @ np.linalg.pinv(W) - P_true, 2))
        M = W.T @ W + s2 * np.eye(q)
        Minv = np.linalg.inv(M)
        SW = S @ W
        W_new = SW @ np.linalg.inv(s2 * np.eye(q) + Minv @ W.T @ SW)
        s2 = np.trace(S - SW @ Minv @ W_new.T) / p
        W = W_new
    gaps, sub = np.array(gaps), np.array(sub)
    g_end = np.argmax(gaps < 1e-10) if np.any(gaps < 1e-10) else T
    s_end = np.argmax(sub < 1e-13) if np.any(sub < 1e-13) else T
    t = np.arange(1, T + 1)
    ax.loglog(t[:g_end + 1], gaps[:g_end + 1], color=CYCLE[i], lw=1.3,
              label="log-likelihood gap" if i == 0 else None)
    ax.loglog(t[:s_end + 1], sub[:s_end + 1], color=CYCLE[i], lw=1.0, ls="--",
              label="distance to principal subspace" if i == 0 else None)
ax.set_ylim(1e-14, 1e5)
ax.set_xlim(1, T)
ax.set_xlabel(r"EM iteration $t$ (log scale)")
ax.set_title("(b) EM from four random starts", loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.6, 0.0), fontsize=8.5)
save(fig, __file__, "ppca-fit")
print(s2_ml, lam[:5])
