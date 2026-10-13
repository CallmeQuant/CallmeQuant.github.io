"""Consistency of maximum likelihood (note: Wald's consistency argument).
Figure 1: Cauchy location model, average log-likelihood ratio versus its limit -KL, and MLE paths.
Figure 2: two-component normal mixture with a free scale, where the likelihood is unbounded."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1949)


def avg_llr(x, th):
    """n^{-1} sum_i log(p_theta / p_0)(x_i) for the Cauchy location family."""
    return -np.mean(np.log1p((x[:, None] - th[None, :]) ** 2) - np.log1p(x[:, None] ** 2), axis=0)


def cauchy_mle(x):
    grid = np.linspace(-15, 15, 3001)
    g = grid[np.argmax(avg_llr(x, grid))]
    res = optimize.minimize_scalar(lambda t: -avg_llr(x, np.array([t]))[0], bracket=(g - 0.02, g, g + 0.02))
    return res.x


# ---- Figure 1 ----
x = rng.standard_cauchy(5000)
th = np.linspace(-6, 6, 1201)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
for col, n in zip([PALETTE["gold"], PALETTE["green"], PALETTE["blue"]], [10, 100, 1000]):
    ax.plot(th, avg_llr(x[:n], th), color=col, lw=1.4, label=rf"$M_n(\theta)$, $n={n}$")
ax.plot(th, -np.log1p(th**2 / 4), color=PALETTE["ink"], lw=1.6, ls="--",
        label=r"$M(\theta) = -\log(1+\theta^2/4)$")
ax.set_xlim(-6, 6); ax.set_ylim(-2.6, 1.3)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("average log-likelihood ratio")
ax.set_title(r"Criterion and its limit ($\theta_0 = 0$)", loc="left")
ax.legend(loc="upper left", fontsize=8.6)

ax = axes[1]
ns = np.unique(np.round(np.logspace(0.7, np.log10(5000), 70)).astype(int))
for j in range(5):
    xs = x if j == 0 else rng.standard_cauchy(5000)
    est = [cauchy_mle(xs[:n]) for n in ns]
    ax.plot(ns, est, color=CYCLE[j], lw=1.2)
ax.axhline(0, color=PALETTE["ink"], lw=0.8)
ax.set_xscale("log"); ax.set_ylim(-3, 3)
ax.set_xlabel(r"sample size $n$ (log scale)"); ax.set_ylabel(r"$\hat\theta_n$")
ax.set_title("Five paths of the global maximizer", loc="left")
save(fig, __file__, "cauchy-criterion")

# ---- Figure 2 ----
# Model: 0.5 N(0,1) + 0.5 N(mu, sigma^2); data from mu = 2, sigma = 1.
n = 200
z = rng.random(n) < 0.5
y = np.where(z, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))


def loglik(mu, logsig):
    """Log-likelihood computed on the log scale, so that sigma = exp(logsig) may be astronomically small."""
    c = -0.5 * np.log(2 * np.pi) + np.log(0.5)
    a = c - y**2 / 2
    d = np.abs(y - mu)
    with np.errstate(divide="ignore"):
        q = np.exp(np.minimum(2 * np.log(d) - 2 * logsig, 700.0))  # ((y - mu)/sigma)^2, capped
    b = c - logsig - 0.5 * q
    return np.sum(np.logaddexp(a, b))


# local maximum of the mu = 2 curve over the scale (dotted reference line)
lmax = -optimize.minimize_scalar(lambda t: -loglik(2.0, t), bounds=(-2.0, 1.5), method="bounded").fun
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharey=True)
for ax, (lo, npts) in zip(axes, [(-8, 600), (-500, 2000)]):
    ls = np.linspace(lo, 1.5, npts)
    ax.plot(ls, [loglik(2.0, t) for t in ls], color=PALETTE["blue"], label=r"$\mu = 2$ (true value)")
    ax.plot(ls, [loglik(y[0], t) for t in ls], color=PALETTE["red"], label=r"$\mu = Y_1$")
    ax.axhline(lmax, color=PALETTE["grey"], lw=0.8, ls=":")
    ax.set_xlabel(r"$\log \sigma$")
axes[0].set_ylabel(r"log-likelihood $\ell_n(\mu,\sigma)$")
axes[0].set_title(r"Near the true scale", loc="left")
axes[1].set_title(r"Down to $\sigma = e^{-500}$", loc="left")
axes[0].legend(loc="upper left")
fig.suptitle(r"Model $\frac{1}{2} N(0,1) + \frac{1}{2} N(\mu,\sigma^2)$, data with $\mu = 2$, $\sigma = 1$, $n = 200$",
             x=0.02, ha="left", fontsize=11.5)
save(fig, __file__, "mixture-degeneracy")
print("local max (mu = 2)", lmax, " ll(Y1, e^-500)", loglik(y[0], -500.0), "crossing", ls[np.argmax(np.array([loglik(y[0], t) for t in ls]) < lmax)])
