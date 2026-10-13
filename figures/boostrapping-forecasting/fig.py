"""Markov-chain bootstrap (Willemain, Smart and Schwarz) for intermittent demand on a simulated series
(note: bootstrapping intermittent demand with a Markov chain). Left: the last 120 periods of the history.
Right: distribution of total demand over a 12-period lead time: truth, Markov bootstrap, i.i.d. bootstrap."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(42)
p01, p11 = 0.10, 0.60            # P(demand | no demand before), P(demand | demand before)
size_mean = 3.0                  # demand size = 1 + Poisson(3)
n_hist, L, B = 730, 12, 20000


def simulate(n, state, rng):
    y = np.zeros(n, dtype=int)
    for t in range(n):
        state = int(rng.random() < (p11 if state else p01))
        if state:
            y[t] = 1 + rng.poisson(size_mean)
    return y


# history, started from the stationary distribution of the occurrence chain
pi1 = p01 / (p01 + 1 - p11)
y = simulate(n_hist, int(rng.random() < pi1), rng)
last = int(y[-1] > 0)

# true lead-time demand distribution given the last observed state
truth = np.array([simulate(L, last, rng).sum() for _ in range(B)])

# Markov bootstrap with jittering
s = (y > 0).astype(int)
n_ij = np.zeros((2, 2))
np.add.at(n_ij, (s[:-1], s[1:]), 1)
P = n_ij / n_ij.sum(axis=1, keepdims=True)
sizes = y[y > 0]


def wss(rng):
    state, total = last, 0
    for _ in range(L):
        state = int(rng.random() < P[state, 1])
        if state:
            x = sizes[rng.integers(sizes.size)]
            xj = 1 + int(np.trunc(x + np.sqrt(x) * rng.standard_normal()))
            total += xj if xj > 0 else x
    return total


boot_markov = np.array([wss(rng) for _ in range(B)])
boot_iid = y[rng.integers(0, n_hist, size=(B, L))].sum(axis=1)

print("estimated P:", P.round(3), "last state", last, "nonzero periods", sizes.size)
for name, d in [("truth", truth), ("markov", boot_markov), ("iid", boot_iid)]:
    print(name, "mean %.2f sd %.2f P0 %.3f q90 %d q95 %d" % (d.mean(), d.std(), np.mean(d == 0),
                                                         np.quantile(d, 0.9), np.quantile(d, 0.95)))

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
t = np.arange(n_hist - 120, n_hist) + 1
yy = y[-120:]
ax.vlines(t[yy > 0], 0, yy[yy > 0], color=PALETTE["blue"], lw=1.4)
ax.plot(t[yy > 0], yy[yy > 0], "o", color=PALETTE["blue"], ms=2.6)
ax.plot(t[yy == 0], np.zeros((yy == 0).sum()), "|", color=PALETTE["grey"], ms=3)
ax.set_xlim(t[0] - 1, t[-1] + 1)
ax.set_ylim(-0.4, max(yy) + 1)
ax.set_xlabel("period $t$"); ax.set_ylabel("demand $y_t$")
ax.set_title("(a) last 120 periods of the history", loc="left")

ax = axes[1]
grid = np.arange(0, 61)
for d, lab, col, ls in [(truth, "true distribution", PALETTE["ink"], "-"),
                        (boot_markov, "Markov bootstrap", PALETTE["red"], "-"),
                        (boot_iid, "i.i.d. bootstrap", PALETTE["gold"], (0, (4, 2)))]:
    ecdf = np.searchsorted(np.sort(d), grid, side="right") / d.size
    ax.step(grid, ecdf, where="post", color=col, lw=1.5, ls=ls, label=lab)
ax.axhline(0.95, color=PALETTE["grey"], lw=0.8, ls=(0, (1, 2)))
ax.text(60, 0.93, "0.95", ha="right", va="top", fontsize=9, color=PALETTE["grey"])
ax.set_xlim(0, 60); ax.set_ylim(0, 1.02)
ax.set_xlabel(r"total demand over $L = 12$ periods")
ax.set_ylabel("distribution function")
ax.set_title("(b) lead-time demand", loc="left")
ax.legend(loc="center right")
save(fig, __file__, "lead-time-demand")
