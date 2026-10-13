"""Seed-to-seed variability: random initialization in a nonconvex fit, random train/test splits,
and the optimism of reporting the best of k seeds (note: random seeds)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()

# ---- Figure 1: frequency estimation by gradient descent from random starting values
rng = np.random.default_rng(101)
n = 100
x = rng.uniform(0, 10, n)
y = np.sin(1.3 * x) + rng.normal(0, 0.3, n)
loss = lambda w: np.mean((y[None, :] - np.sin(np.outer(w, x))) ** 2, axis=1)
grad = lambda w: -2 * np.mean((y[None, :] - np.sin(np.outer(w, x))) * np.cos(np.outer(w, x)) * x[None, :], axis=1)

w = np.random.default_rng(7).uniform(0.2, 3.0, 300)      # one starting value per seed
w0 = w.copy()
for _ in range(3000):
    w = w - 0.004 * grad(w)
ws = np.linspace(0.05, 3.2, 2000)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ax.plot(ws, loss(ws), color=PALETTE["ink"], lw=1.2)
ends = np.round(w, 3)
vals, counts = np.unique(np.round(w, 2), return_counts=True)
ax.scatter(vals, loss(vals), s=12 + 1.6 * counts, color=PALETTE["red"], alpha=0.8, zorder=3,
           label="end points of gradient descent\n(area grows with the number\nof seeds ending there)")
ax.set_xlim(0.05, 3.2); ax.set_ylim(0, 1.25)
ax.set_xlabel(r"frequency $\omega$")
ax.set_ylabel(r"least-squares criterion $L_n(\omega)$")
ax.set_title("Nonconvex least squares, 300 random starts", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.52, 0.66), fontsize=8.6, markerscale=0.6)

ax = axes[1]
best = ws[np.argmin(loss(ws))]
hit = np.abs(w - best) < 0.02
ax.scatter(w0[hit], w[hit], s=6, color=PALETTE["blue"], label=rf"reaches global minimum ({hit.mean():.0%})")
ax.scatter(w0[~hit], w[~hit], s=6, color=PALETTE["red"], label="stops at a local minimum")
ax.set_xlabel(r"starting value $\omega^{(0)}$ (set by the seed)")
ax.set_ylabel(r"final value $\hat\omega$")
ax.set_xlim(0.1, 3.1); ax.set_ylim(0, 3.3)
ax.set_title("Output as a function of the seed", loc="left")
ax.legend(loc="upper left", fontsize=8.6, markerscale=2)
save(fig, __file__, "nonconvex-starts")

# ---- Figure 2: test R^2 across random splits, and best-of-k optimism
rng = np.random.default_rng(202)
n, p = 60, 8
X = rng.standard_normal((n, p))
beta = np.array([1.0, -0.8, 0.6, 0.5, 0.0, 0.0, 0.3, 0.0])
yv = X @ beta + rng.normal(0, 1.2, n)


def test_r2(seed):
    idx = np.random.default_rng(seed).permutation(n)
    tr, te = idx[:45], idx[45:]
    A = np.column_stack([np.ones(45), X[tr]])
    coef = np.linalg.lstsq(A, yv[tr], rcond=None)[0]
    pred = np.column_stack([np.ones(n - 45), X[te]]) @ coef
    return 1 - np.sum((yv[te] - pred) ** 2) / np.sum((yv[te] - yv[te].mean()) ** 2)


r2 = np.array([test_r2(s) for s in range(5000)])
mu, sd = r2.mean(), r2.std()

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axes[0]
ax.hist(r2, bins=60, color=PALETTE["light"], edgecolor="white", lw=0.4, density=True)
ax.axvline(mu, color=PALETTE["blue"], lw=1.3, label=rf"mean over seeds {mu:.2f}")
ax.axvline(test_r2(42), color=PALETTE["red"], lw=1.3, ls="--", label=f"seed 42: {test_r2(42):.2f}")
ax.set_xlabel(r"test $R^2$ (45 training, 15 test points)"); ax.set_ylabel("density")
ax.set_title(rf"One data set, 5000 split seeds (sd {sd:.2f})", loc="left")
ax.set_xlim(-0.6, 1.0)
ax.legend(loc="upper left", fontsize=8.6)

ax = axes[1]
ks = np.unique(np.round(np.geomspace(1, 200, 25)).astype(int))
B = 4000
rb = np.random.default_rng(303)
emax = [np.mean(np.max(rb.choice(r2, size=(B, k)), axis=1)) for k in ks]
ax.plot(ks, emax, "o-", color=PALETTE["red"], ms=3.5, lw=1.2, label=r"expected best of $k$ seeds")
ax.plot(ks, mu + sd * np.sqrt(2 * np.log(ks)), color=PALETTE["grey"], lw=1.2, ls="--",
        label=r"$\mu + \sigma\sqrt{2\log k}$")
ax.axhline(mu, color=PALETTE["blue"], lw=1.2, label=r"$\mu$: what a new split delivers")
ax.set_xscale("log"); ax.set_ylim(0.2, 1.3)
ax.axhline(1, color=PALETTE["ink"], lw=0.6, ls=":")
ax.text(1.05, 1.02, r"$R^2 \leq 1$", fontsize=9, va="bottom")
ax.set_xlabel(r"number of seeds tried, $k$"); ax.set_ylabel(r"reported test $R^2$")
ax.set_title("Optimism from choosing the seed", loc="left")
ax.legend(loc="lower right", fontsize=8.6)
save(fig, __file__, "split-seeds")
print(mu, sd, hit.mean())
