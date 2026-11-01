"""Figures for "Markov chains, dynamic programming and fixed points" (mean-field games, Part 0).

Figure 1  empirical-measure.svg   empirical measure of N independent chains and its L1 error
Figure 2  backward-forward.svg    optimal policy (backward induction) and the state distribution it generates
Figure 3  fixed-points.svg        best responses in a coordination game; closed-graph correspondences
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import binom
from style import use_style, PALETTE, save

use_style()

# ---------------------------------------------------------------- the example MDP
d, T, eta, kappa, lam = 11, 14, 0.3, 0.5, 0.15
X = np.arange(d)
A = np.array([-1, 0, 1])


def clip(z):
    return int(np.clip(z, 0, d - 1))


p = np.zeros((3, d, d))                      # p[a, x, y] = p(y | x, a)
for k, a in enumerate(A):
    for x in X:
        p[k, x, clip(x + a)] += 1 - eta
        p[k, x, clip(x + a - 1)] += eta / 2
        p[k, x, clip(x + a + 1)] += eta / 2
c = kappa * np.abs(A)[None, :] + lam * ((X - 3) ** 2)[:, None]   # c(x, a)
g = (X - 8.0) ** 2

V = np.zeros((T + 1, d)); V[T] = g
Q = np.zeros((T, d, 3))
for t in range(T - 1, -1, -1):
    Q[t] = c + np.einsum("kxy,y->xk", p, V[t + 1])
    V[t] = Q[t].min(axis=1)
policy = Q.argmin(axis=2)                     # index into A

m0 = np.zeros(d); m0[1:6] = binom.pmf(np.arange(5), 4, 0.5)
mu = np.zeros((T + 1, d)); mu[0] = m0
for t in range(T):
    mu[t + 1] = mu[t] @ p[policy[t], X, :]
J = sum(mu[t] @ c[X, policy[t]] for t in range(T)) + mu[T] @ g
assert abs(J - m0 @ V[0]) < 1e-9

# ---------------------------------------------------------------- Figure 2 (backward-forward)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
act_cmap = ListedColormap([PALETTE["red"], "#eef1f6", PALETTE["blue"]])
ax = axes[0]
ax.imshow(policy.T, origin="lower", cmap=act_cmap, vmin=-0.5, vmax=2.5, aspect="auto",
          extent=(-0.5, T - 0.5, -0.5, d - 0.5))
for t in range(T):
    for x in X:
        sym = {0: r"$\downarrow$", 1: r"$\cdot$", 2: r"$\uparrow$"}[policy[t, x]]
        col = "white" if policy[t, x] != 1 else PALETTE["ink"]
        ax.text(t, x, sym, ha="center", va="center", fontsize=9, color=col)
ax.set_title(r"(a) optimal action $\varphi_t(x)$", loc="left")
ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"state $x$")
ax.set_xticks(range(0, T, 2)); ax.set_yticks(range(0, d, 2)); ax.grid(False)

ax = axes[1]
blues = LinearSegmentedColormap.from_list("b", ["#ffffff", "#c9d5ef", PALETTE["blue"], "#16264d"])
im = ax.imshow(mu.T, origin="lower", cmap=blues, vmin=0, vmax=mu.max(), aspect="auto",
               extent=(-0.5, T + 0.5, -0.5, d - 0.5))
ax.set_title(r"(b) state distribution $\mu_t(x)$", loc="left")
ax.set_xlabel(r"time $t$"); ax.set_xticks(range(0, T + 1, 2)); ax.set_yticks(range(0, d, 2))
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5)
save(fig, __file__, "backward-forward")

# ---------------------------------------------------------------- Figure 1 (empirical measure)
rng = np.random.default_rng(20261101)
muT = mu[T]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
w = 0.38
for k, (N, col) in enumerate([(40, PALETTE["gold"]), (400, PALETTE["green"])]):
    emp = rng.multinomial(N, muT) / N
    ax.bar(X + (k - 0.5) * w, emp, width=w, color=col, alpha=0.9, label=rf"$\bar\mu^N_T$, $N={N}$")
ax.plot(X, muT, "o", color=PALETTE["ink"], ms=4.5, label=r"$\mu_T$")
ax.set_xticks(range(0, d, 2)); ax.set_xlabel(r"state $x$")
ax.set_title("(a) one draw of the empirical measure", loc="left")
ax.legend(loc="upper left"); ax.set_ylim(0, 0.7)

ax = axes[1]
Ns = np.unique(np.round(np.logspace(1, 4, 13)).astype(int))
reps = 4000
err = np.array([np.abs(rng.multinomial(N, muT, size=reps) / N - muT).sum(axis=1).mean() for N in Ns])
ax.loglog(Ns, err, "o", color=PALETTE["blue"], ms=4.5, label=r"Monte Carlo $\mathbb{E}\Vert\bar\mu^N_T-\mu_T\Vert_1$")
ax.loglog(Ns, np.sqrt((d - 1) / Ns), color=PALETTE["red"], label=r"bound $\sqrt{(d-1)/N}$")
clt = np.sqrt(2 / np.pi) * np.sqrt(muT * (1 - muT)).sum()
ax.loglog(Ns, clt / np.sqrt(Ns), "--", color=PALETTE["grey"], label=r"CLT approximation")
ax.set_xlabel(r"number of chains $N$")
ax.set_title(r"(b) $L^1$ error at time $T$", loc="left")
ax.legend(loc="lower left")
save(fig, __file__, "empirical-measure")

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
ax = axes[0]
s = 1 / 3
# player 1 chooses p = P(L) given q; player 2 chooses q given p
ax.plot([0, 0, 1, 1], [0, s, s, 1], color=PALETTE["blue"], lw=2.2, label=r"$\mathrm{BR}_1(q)$")
ax.plot([0, s, s, 1], [0, 0, 1, 1], color=PALETTE["red"], lw=2.2, ls=(0, (5, 2)), label=r"$\mathrm{BR}_2(p)$")
for (x0, y0) in [(0, 0), (s, s), (1, 1)]:
    ax.plot(x0, y0, "o", ms=8, mfc="white", mec=PALETTE["ink"], mew=1.4, zorder=5)
ax.annotate(r"$(1/3,\,1/3)$", (s, s), xytext=(0.47, 0.18), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=PALETTE["grey"], lw=0.8))
ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06)
ax.set_xlabel(r"$p$ = probability that player 1 plays L")
ax.set_ylabel(r"$q$ = probability that player 2 plays L")
ax.set_title("(a) three Nash equilibria", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
ax.set_aspect("equal")

ax = axes[1]
ax.plot([0, 1], [0, 1], ":", color=PALETTE["grey"], lw=1.2)
ax.plot([0, 0.5], [0.8, 0.8], color=PALETTE["blue"], lw=2.2, label=r"$F(1/2)=[0.2,0.8]$")
ax.plot([0.5, 1], [0.2, 0.2], color=PALETTE["blue"], lw=2.2)
ax.plot([0.5, 0.5], [0.2, 0.8], color=PALETTE["blue"], lw=2.2)
ax.plot(0.5, 0.5, "o", ms=8, mfc="white", mec=PALETTE["ink"], mew=1.4, zorder=5)
ax.plot([0.5, 0.5], [0.2, 0.8], "o", ms=6, color=PALETTE["red"], zorder=6, label=r"$G(1/2)=\{0.2,0.8\}$")
ax.text(0.55, 0.46, "fixed point of $F$", fontsize=9.5, color=PALETTE["ink"])
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$F(x)$, $G(x)$")
ax.set_title("(b) why the values must be convex", loc="left")
ax.legend(loc="upper left")
save(fig, __file__, "fixed-points")

if __name__ == "__main__":
    print("J =", round(J, 6), " <m0, V0> =", round(m0 @ V[0], 6))
    print("E||mu_N - mu||_1 at N=10, 10^4:", err[0], err[-1], " CLT constant:", clt)
