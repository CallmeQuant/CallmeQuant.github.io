"""AdaBoost with decision stumps on a disc-shaped class (note: AdaBoost as stagewise
minimization of the exponential loss). Produces two figures:
  rounds.svg  - decision region after M rounds; training error, test error and the bound
  margins.svg - cumulative distribution of normalized training margins for several M
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(2718)
R2 = 0.55                     # class +1 is the disc x1^2 + x2^2 < R2
N_TRAIN, N_TEST, M = 200, 5000, 400
SNAPSHOTS = [5, 50, 400]


def sample(n):
    x = rng.uniform(-1, 1, size=(n, 2))
    y = np.where((x ** 2).sum(1) < R2, 1, -1)
    return x, y


def fit_stump(x, y, w):
    """Weighted decision stump h(x) = s * sign(x_j - t): returns (j, t, s, weighted error)."""
    best = (0, 0.0, 1, np.inf)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j])
        xs, ys, ws = x[order, j], y[order], w[order]
        # error of s = +1 (predict +1 to the right of t) when the split is after position k
        left_pos = np.concatenate([[0.0], np.cumsum(ws * (ys == 1))])     # +1 points on the left
        right_neg = np.concatenate([[0.0], np.cumsum((ws * (ys == -1))[::-1])])[::-1]
        err = left_pos + right_neg
        cuts = np.concatenate([[xs[0] - 1e-9], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1e-9]])
        for s, e in ((1, err), (-1, 1.0 - err)):
            k = int(np.argmin(e))
            if e[k] < best[3]:
                best = (j, cuts[k], s, e[k])
    return best


def stump_predict(x, j, t, s):
    return s * np.where(x[:, j] > t, 1, -1)


def adaboost(x, y, rounds):
    n = len(y)
    w = np.full(n, 1.0 / n)
    stumps, alphas, Z = [], [], []
    for _ in range(rounds):
        j, t, s, eps = fit_stump(x, y, w)
        alpha = 0.5 * np.log((1 - eps) / eps)
        h = stump_predict(x, j, t, s)
        w = w * np.exp(-alpha * y * h)
        Z.append(w.sum())
        w /= w.sum()
        stumps.append((j, t, s)); alphas.append(alpha)
    return stumps, np.array(alphas), np.array(Z)


def score(x, stumps, alphas):
    """Matrix of F_m(x) for m = 1..len(alphas) (rows = rounds)."""
    H = np.array([stump_predict(x, *st) for st in stumps])
    return np.cumsum(alphas[:, None] * H, axis=0)


xtr, ytr = sample(N_TRAIN)
xte, yte = sample(N_TEST)
stumps, alphas, Z = adaboost(xtr, ytr, M)
Ftr, Fte = score(xtr, stumps, alphas), score(xte, stumps, alphas)
train_err = (np.sign(Ftr) != ytr).mean(1)
test_err = (np.sign(Fte) != yte).mean(1)
bound = np.cumprod(Z)
m = np.arange(1, M + 1)

# ---- Figure 1: decision region and error curves -------------------------------------
fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1, 1.35]})
g = np.linspace(-1, 1, 301)
G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
Fg = score(G, stumps[:SNAPSHOTS[1]], alphas[:SNAPSHOTS[1]])[-1].reshape(301, 301)
a0.contourf(g, g, np.sign(Fg), levels=[-2, 0, 2],
            colors=["#f3e3dd", "#dfe5f2"])
a0.contour(g, g, Fg, levels=[0], colors=PALETTE["ink"], linewidths=0.9)
th = np.linspace(0, 2 * np.pi, 400)
a0.plot(np.sqrt(R2) * np.cos(th), np.sqrt(R2) * np.sin(th), ls="--", lw=1.0, color=PALETTE["grey"])
a0.scatter(*xtr[ytr == 1].T, s=11, color=PALETTE["blue"], label=r"$y=+1$")
a0.scatter(*xtr[ytr == -1].T, s=11, color=PALETTE["red"], marker="x", lw=0.9, label=r"$y=-1$")
a0.set_aspect("equal"); a0.set_xlim(-1, 1); a0.set_ylim(-1, 1)
a0.set_xticks([-1, 0, 1]); a0.set_yticks([-1, 0, 1]); a0.grid(False)
a0.set_xlabel(r"$x_1$"); a0.set_ylabel(r"$x_2$")
a0.set_title(rf"$\mathrm{{sign}}\,F_M$ after $M = {SNAPSHOTS[1]}$ stumps", loc="left")

a1.plot(m, bound, color=PALETTE["gold"], label=r"bound $Z_1 Z_2 \cdots Z_M$")
a1.plot(m, train_err, color=PALETTE["blue"], label="training error")
a1.plot(m, test_err, color=PALETTE["red"], label="test error")
a1.set_xscale("log")
a1.set_ylim(0, 1.02); a1.set_xlim(1, M)
a1.set_xlabel(r"number of rounds $M$")
a1.set_title("error rates and the exponential-loss bound", loc="left")
a1.legend(loc="upper right")
save(fig, __file__, "rounds")

# ---- Figure 2: margin distributions --------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for M_, col in zip(SNAPSHOTS, [PALETTE["grey"], PALETTE["green"], PALETTE["violet"]]):
    marg = np.sort(ytr * Ftr[M_ - 1] / alphas[:M_].sum())
    ax.step(np.concatenate([[-1], marg, [1]]),
            np.concatenate([[0], np.arange(1, N_TRAIN + 1) / N_TRAIN, [1]]),
            where="post", color=col,
            label=rf"$M = {M_}$ (train err {train_err[M_ - 1]:.3f})")
ax.axvline(0, color=PALETTE["ink"], lw=0.7)
ax.set_xlim(-0.3, 0.6); ax.set_ylim(0, 1.02)
ax.set_xlabel(r"normalized margin $y_i f_M(x_i)$")
ax.set_ylabel("fraction of training points")
ax.legend(loc="lower right")
save(fig, __file__, "margins")
print("final train err", train_err[-1], "test err", test_err[[4, 49, 399]], "first zero", np.argmax(train_err == 0) + 1)
print("eps range", (1 / (1 + np.exp(2 * alphas))).min(), (1 / (1 + np.exp(2 * alphas))).max())
