"""Cover's universal portfolio on a simulated two-stock market (note: universal portfolios)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()

# Two stocks sharing a random-walk factor F, with a mean-reverting spread s:
# log price A = F + s + 0.0002 t, log price B = F - s.
rng = np.random.default_rng(5)
n = 750
F = np.cumsum(rng.normal(0.0, 0.012, n + 1))
s = np.zeros(n + 1)
for t in range(1, n + 1):
    s[t] = 0.9 * s[t - 1] + rng.normal(0.0, 0.04)
logA = F + s + 0.0002 * np.arange(n + 1)
logB = F - s
X = np.exp(np.column_stack([np.diff(logA), np.diff(logB)]))      # price relatives x_t

# Constant rebalanced portfolios b = (w, 1 - w) on a fine grid; uniform prior on w.
w = np.linspace(0, 1, 4001)
logS = np.vstack([np.zeros_like(w), np.cumsum(np.log(np.outer(X[:, 0], w) + np.outer(X[:, 1], 1 - w)), axis=0)])
# Universal portfolio: weight of A at time t is the S_{t-1}-weighted average of w.
W = np.exp(logS - logS.max(axis=1, keepdims=True))
w_hat = (W * w).sum(1) / W.sum(1)                                  # w_hat[t] is used in period t+1
log_up = np.concatenate([[0.0], np.cumsum(np.log(w_hat[:-1] * X[:, 0] + (1 - w_hat[:-1]) * X[:, 1]))])
log_mix = np.log(W.mean(1)) + logS.max(axis=1)                     # log of the prior average of S_t(b)
assert np.max(np.abs(log_up - log_mix)) < 1e-6                      # Lemma 1, numerically
k_star = logS[-1].argmax(); w_star = w[k_star]
log_bh = np.log(0.5 * np.exp(logA - logA[0]) + 0.5 * np.exp(logB - logB[0]))
t = np.arange(n + 1)

# ---- Figure 1: wealth paths and final wealth of every CRP
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.plot(t, logA - logA[0], color=PALETTE["grey"], lw=0.9, label="stock A")
ax.plot(t, logB - logB[0], color=PALETTE["violet"], lw=0.9, alpha=0.8, label="stock B")
ax.plot(t, log_bh, color=PALETTE["gold"], lw=1.2, label=r"buy and hold, half in each")
ax.plot(t, logS[:, k_star], color=PALETTE["green"], lw=1.4, label=rf"best CRP in hindsight, $w^* = {w_star:.2f}$")
ax.plot(t, log_up, color=PALETTE["blue"], lw=1.6, label="universal portfolio")
ax.axhline(0, color=PALETTE["ink"], lw=0.5)
ax.set_xlim(0, n); ax.set_ylim(-0.58, 0.85); ax.set_xlabel("period $t$"); ax.set_ylabel(r"log wealth")
ax.set_title("Wealth over time (initial wealth 1)", loc="left")
ax.legend(loc="upper left", fontsize=8.4)

ax = axes[1]
ax.plot(w, np.exp(logS[-1]), color=PALETTE["green"], lw=1.4, label=r"$S_n(b)$, CRP with weight $w$ on A")
ax.axhline(np.exp(log_up[-1]), color=PALETTE["blue"], lw=1.3, label=r"$\hat S_n = \int_0^1 S_n(b)\,\mathrm{d}w$")
ax.plot([0, 1], np.exp(logS[-1, [0, -1]]), "o", color=PALETTE["ink"], ms=4)
ax.annotate("all in B", (0, np.exp(logS[-1, 0])), xytext=(0.03, np.exp(logS[-1, 0]) - 0.12), fontsize=9)
ax.annotate("all in A", (1, np.exp(logS[-1, -1])), xytext=(0.97, np.exp(logS[-1, -1]) - 0.12), fontsize=9, ha="right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.75, 2.2)
ax.set_xlabel("weight $w$ on stock A"); ax.set_ylabel(f"final wealth after $n = {n}$ periods")
ax.set_title("Every constant rebalancing rule", loc="left")
ax.legend(loc="upper center", fontsize=8.4)
save(fig, __file__, "wealth")

# ---- Figure 2: portfolio weights and regret over time
best_t = logS.max(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
ax = axes[0]
ax.plot(t[1:], w[logS[1:].argmax(axis=1)], color=PALETTE["green"], lw=1.0, label="best CRP weight up to time $t$")
ax.plot(t[:-1] + 1, w_hat[:-1], color=PALETTE["blue"], lw=1.5, label=r"universal portfolio weight $\hat w_t$")
ax.axhline(w_star, color=PALETTE["ink"], lw=0.6, ls="--")
ax.set_xlim(0, n); ax.set_ylim(0, 1)
ax.set_xlabel("period $t$"); ax.set_ylabel("weight on stock A")
ax.set_title("Portfolio weights", loc="left")
ax.legend(loc="upper right", fontsize=8.4)

ax = axes[1]
ax.plot(t, best_t - log_up, color=PALETTE["blue"], lw=1.4, label=r"$\log S_t^* - \log \hat S_t$")
ax.plot(t, np.log(t + 1) + 1, color=PALETTE["red"], lw=1.2, ls="--", label=r"bound $(m-1)\log(t+1) + 1$, $m = 2$")
ax.set_xlim(0, n); ax.set_ylim(0, 8.5)
ax.set_xlabel("period $t$"); ax.set_ylabel("regret (log scale of wealth)")
ax.set_title("Regret against the best CRP", loc="left")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8.4)
save(fig, __file__, "weights-regret")
print(w_star, logA[-1] - logA[0], logB[-1] - logB[0], logS[-1, k_star], log_up[-1], log_bh[-1], best_t[-1] - log_up[-1])
