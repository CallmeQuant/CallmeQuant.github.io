"""Rao-Blackwellization in a two-block Gibbs sampler (note: conditioning out the noise).
Data augmentation for Student t_3: Y | X ~ Gamma((nu+1)/2, rate (nu+X^2)/2), X | Y ~ N(0, 1/Y).
Left: density of X from one chain, kernel estimate versus Rao-Blackwellized estimate.
Right: sampling distribution of two estimators of P(X > 3) over independent chains."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1990)
NU, N, A = 3.0, 500, 3.0


def gibbs(n_iter, n_chains):
    """Returns arrays (X, Y) of shape (n_chains, n_iter); X_0 drawn from the target (stationary start)."""
    X = np.empty((n_chains, n_iter)); Y = np.empty((n_chains, n_iter))
    x = rng.standard_t(NU, n_chains)
    for t in range(n_iter):
        y = rng.gamma((NU + 1) / 2, 2 / (NU + x**2))
        x = rng.standard_normal(n_chains) / np.sqrt(y)
        Y[:, t], X[:, t] = y, x
    return X, Y


fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
X, Y = gibbs(N, 1)
x, y = X[0], Y[0]
grid = np.linspace(-8, 8, 801)
kde = stats.gaussian_kde(x)(grid)
rb = np.mean(stats.norm.pdf(grid[:, None], scale=1 / np.sqrt(y[None, :])), axis=1)
ax.semilogy(grid, stats.t.pdf(grid, NU), color=PALETTE["ink"], ls="--", lw=1.4, label=r"target $t_3$ density")
ax.semilogy(grid, kde, color=PALETTE["grey"], lw=1.4, label="kernel estimate from $X_t$")
ax.semilogy(grid, rb, color=PALETTE["blue"], lw=1.6, label=r"$\frac{1}{N}\sum_t \pi(x \mid Y_t)$")
ax.set_ylim(1e-4, 0.8); ax.set_xlim(-8, 8)
ax.set_xlabel(r"$x$"); ax.set_ylabel("density (log scale)")
ax.set_title(rf"One chain, $N = {N}$", loc="left")
ax.legend(loc="lower center", fontsize=8.6)

ax = axes[1]
X, Y = gibbs(N, 4000)
crude = np.mean(X > A, axis=1)
rbp = np.mean(stats.norm.sf(A * np.sqrt(Y)), axis=1)
truth = stats.t.sf(A, NU)
bins = (np.arange(0, 37) - 0.5) / N   # one bin per attainable value k/N of the crude estimator
ax.hist(crude, bins=bins, color=PALETTE["grey"], alpha=0.55, label=rf"crude: sd {crude.std():.4f}")
ax.hist(rbp, bins=bins, color=PALETTE["blue"], alpha=0.75, label=rf"Rao–Blackwell: sd {rbp.std():.4f}")
ax.axvline(truth, color=PALETTE["ink"], lw=1.0, ls="--")
ax.set_xlabel(r"estimate of $\mathrm{P}(X > 3) = %.4f$" % truth)
ax.set_ylabel("number of chains")
ax.set_title(r"4000 chains, $N = %d$ each" % N, loc="left")
ax.legend(loc="upper right", fontsize=8.6)
save(fig, __file__, "rb-gibbs")
print("truth", truth, "crude mean/sd", crude.mean(), crude.std(), "rb", rbp.mean(), rbp.std())
