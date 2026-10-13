"""Gibbs sampling and random-walk Metropolis for a normal model with semi-conjugate prior
(note: Markov chains for posterior sampling)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()

# Data and prior: y_i ~ N(mu, 1/tau); mu ~ N(0, 10^2); tau ~ Gamma(shape 2, rate 1).
rng = np.random.default_rng(2024)
n = 12
y = rng.normal(1.5, 0.8, n)
a0, b0, kappa0 = 2.0, 1.0, 1 / 100


def log_post(mu, eta):
    """Log posterior density of (mu, eta = log tau), up to an additive constant."""
    tau = np.exp(eta)
    ss = ((y - mu) ** 2).sum()
    return (n / 2 + a0) * eta - tau * (b0 + ss / 2) - kappa0 * mu**2 / 2


def gibbs(T, mu, tau, rng):
    out = np.empty((2 * T + 1, 2)); out[0] = mu, np.log(tau); k = 1
    for _ in range(T):
        prec = tau * n + kappa0
        mu = rng.normal(tau * y.sum() / prec, 1 / np.sqrt(prec)); out[k] = mu, np.log(tau); k += 1
        tau = rng.gamma(a0 + n / 2, 1 / (b0 + ((y - mu) ** 2).sum() / 2)); out[k] = mu, np.log(tau); k += 1
    return out


def rwm(T, x, h, rng):
    out = np.empty((T + 1, 2)); out[0] = x; lp = log_post(*x); acc = 0
    for t in range(T):
        prop = x + h * rng.standard_normal(2)
        lq = log_post(*prop)
        if np.log(rng.random()) < lq - lp:
            x, lp = prop, lq; acc += 1
        out[t + 1] = x
    return out, acc / T


# ---- Figure 1: first moves of each sampler over posterior contours
M, E = np.meshgrid(np.linspace(-1.2, 3.6, 300), np.linspace(-2.4, 1.4, 300))
LP = np.vectorize(log_post)(M, E)
P = np.exp(LP - LP.max())
start = np.array([-0.3, -2.0])
g = gibbs(25, start[0], np.exp(start[1]), np.random.default_rng(1))
r, _ = rwm(50, start, 0.35, np.random.default_rng(2))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
for ax, path, col, title in [(axes[0], g, PALETTE["blue"], "Gibbs sampler, 25 sweeps"),
                             (axes[1], r, PALETTE["red"], "Random-walk Metropolis, 50 steps")]:
    ax.contour(M, E, P, levels=[0.02, 0.1, 0.3, 0.6, 0.9], colors=PALETTE["grey"], linewidths=0.8)
    ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=1.0, alpha=0.85)
    ax.plot(path[:, 0], path[:, 1], "o", color=col, ms=2.8)
    ax.plot(*start, marker="s", color=PALETTE["ink"], ms=5)
    ax.set_title(title, loc="left")
    ax.set_xlabel(r"$\mu$")
    ax.set_xlim(-1.2, 3.6); ax.set_ylim(-2.4, 1.4)
axes[0].set_ylabel(r"$\eta = \log \tau$")
save(fig, __file__, "sampler-paths")

# ---- Figure 2: traces and autocorrelations of mu for three proposal scales
T = 4000
rng = np.random.default_rng(7)
hs = [0.05, 0.5, 5.0]
cols = [PALETTE["gold"], PALETTE["blue"], PALETTE["red"]]
runs = [rwm(T, np.array([1.5, 0.0]), h, rng) for h in hs]


def acf(x, K):
    x = x - x.mean(); v = (x * x).mean()
    return np.array([(x[: len(x) - k] * x[k:]).mean() / v for k in range(K + 1)])


fig = plt.figure(figsize=(7.8, 4.4))
gs = fig.add_gridspec(3, 2, width_ratios=[2.1, 1])
ax_acf = fig.add_subplot(gs[:, 1])
for i, (h, col, (path, acc)) in enumerate(zip(hs, cols, runs)):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(np.arange(1001), path[:1001, 0], color=col, lw=0.8)
    ax.set_ylim(0.2, 2.8); ax.set_yticks([0.5, 1.5, 2.5]); ax.set_xlim(0, 1000)
    ax.set_title(rf"$h = {h}$, acceptance rate {acc:.2f}", loc="left", fontsize=10.5)
    ax.set_ylabel(r"$\mu_t$")
    if i < 2:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("iteration $t$")
    ax_acf.plot(np.arange(61), acf(path[1000:, 0], 60), color=col, label=rf"$h = {h}$")
ax_acf.axhline(0, color=PALETTE["ink"], lw=0.6)
ax_acf.set_xlabel("lag $k$"); ax_acf.set_ylabel(r"autocorrelation of $\mu_t$")
ax_acf.set_title("Autocorrelation", loc="left", fontsize=10.5)
ax_acf.legend(loc="upper right")
save(fig, __file__, "step-size")
