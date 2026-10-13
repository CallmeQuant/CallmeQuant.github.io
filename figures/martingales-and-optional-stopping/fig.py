"""Gambler's ruin paths and likelihood-ratio martingales with Ville's threshold
(note: martingales and optional stopping)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1939)

# ---------- Gambler's ruin: simple random walk from 0 stopped at -a or b ----------
a, b, n_show, n_sim = 6, 4, 14, 100_000
def run(rng):
    s, path = 0, [0]
    while -a < s < b:
        s += 1 if rng.random() < 0.5 else -1
        path.append(s)
    return np.array(path)
paths = [run(rng) for _ in range(n_show)]
# Monte Carlo for exit probabilities and mean exit time (vectorized)
S = np.zeros(n_sim, dtype=int); tau = np.zeros(n_sim, dtype=int); alive = np.ones(n_sim, bool)
while alive.any():
    S[alive] += rng.choice([-1, 1], size=alive.sum())
    tau[alive] += 1
    alive &= (S > -a) & (S < b)
p_hat, tau_hat = (S == b).mean(), tau.mean()

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
for p in paths:
    col = PALETTE["blue"] if p[-1] == b else PALETTE["red"]
    ax.step(np.arange(len(p)), p, where="post", color=col, lw=0.9, alpha=0.75)
    ax.plot(len(p) - 1, p[-1], "o", color=col, ms=3.5)
ax.axhline(b, color=PALETTE["ink"], lw=0.9); ax.axhline(-a, color=PALETTE["ink"], lw=0.9)
ax.set_xlim(0, 80); ax.set_ylim(-a - 0.8, b + 0.8)
ax.set_yticks([-6, -4, -2, 0, 2, 4])
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$S_{\tau \wedge n}$")
ax.set_title("Walks stopped at $-6$ or $4$", loc="left")

# ---------- Likelihood ratio martingale N(theta,1) vs N(0,1) ----------
theta, N, alpha = 0.5, 400, 0.05
n = np.arange(1, N + 1)
def log10_lr(mean, m):
    X = rng.normal(mean, 1.0, size=(m, N))
    return (theta * np.cumsum(X, 1) - n * theta ** 2 / 2) / np.log(10)
L0 = log10_lr(0.0, 20_000)
cross = (L0.max(1) >= np.log10(1 / alpha)).mean()
L1 = log10_lr(theta, 8)
ax = axes[1]
for i in range(40):
    ax.plot(n, L0[i], color=PALETTE["grey"], lw=0.6, alpha=0.8)
for i in range(8):
    ax.plot(n, L1[i], color=PALETTE["green"], lw=0.8)
ax.axhline(np.log10(1 / alpha), color=PALETTE["red"], lw=1.2, ls="--")
ax.text(395, np.log10(1 / alpha) + 0.25, r"$L_n = 1/\alpha = 20$", color=PALETTE["red"], fontsize=10, ha="right")
ax.plot([], [], color=PALETTE["grey"], label="data from the null $N(0,1)$")
ax.plot([], [], color=PALETTE["green"], label=r"data from $N(0.5, 1)$")
ax.set_xlim(0, N); ax.set_ylim(-8, 8)
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$\log_{10} L_n$")
ax.set_title(rf"Fraction of null paths ever above $20$: {cross:.3f}", loc="left")
ax.legend(loc="upper right", fontsize=8.8)
save(fig, __file__, "martingale-paths")
print(p_hat, tau_hat, cross)
