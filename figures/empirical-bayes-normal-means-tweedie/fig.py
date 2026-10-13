"""Empirical Bayes for p = 1000 normal means with a two-component prior: the marginal density of X
and its log-polynomial estimate, and the estimated rules (linear/James-Stein and Tweedie) against the
oracle posterior mean (note: empirical Bayes)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()
p, w1, mu1, s1 = 1000, 0.3, 3.0, 0.5     # prior G = 0.7 delta_0 + 0.3 N(3, 0.5^2); noise sd 1


def oracle(x):
    """E[theta | X = x] under G."""
    f0 = (1 - w1) * norm.pdf(x)
    v = 1 + s1**2
    f1 = w1 * norm.pdf(x, mu1, np.sqrt(v))
    m1 = mu1 + s1**2 / v * (x - mu1)
    return f1 * m1 / (f0 + f1)


def marginal(x):
    return (1 - w1) * norm.pdf(x) + w1 * norm.pdf(x, mu1, np.sqrt(1 + s1**2))


def james_stein(x):
    xb = x.mean()
    b = min(1.0, (x.size - 3) / np.sum((x - xb) ** 2))
    return xb + (1 - b) * (x - xb)


def tweedie(x, at, degree=5, nbins=60):
    """x + (log f)'(x), with log f a degree-5 polynomial fitted to histogram counts by Poisson
    regression (Lindsey's method), as in Efron's treatment of Tweedie's formula."""
    edges = np.linspace(x.min() - 1e-9, x.max() + 1e-9, nbins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    y = np.histogram(x, edges)[0].astype(float)
    c, s = mids.mean(), mids.std()
    Z = np.vander((mids - c) / s, degree + 1, increasing=True)
    beta = np.zeros(degree + 1); beta[0] = np.log(y.mean() + 1e-9)
    for _ in range(100):                                   # IRLS for the Poisson GLM
        mu = np.exp(Z @ beta)
        step = np.linalg.solve(Z.T @ (mu[:, None] * Z), Z.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    u = (at - c) / s
    logf = np.vander(u, degree + 1, increasing=True) @ beta
    dlogf = np.vander(u, degree, increasing=True) @ (beta[1:] * np.arange(1, degree + 1)) / s
    f = np.exp(logf) / (x.size * width)
    return at + dlogf, f


def draw(rng):
    z = rng.random(p) < w1
    theta = np.where(z, rng.normal(mu1, s1, p), 0.0)
    return theta, theta + rng.standard_normal(p)


rng = np.random.default_rng(1956)
theta, x = draw(rng)
grid = np.linspace(-3.5, 6.5, 600)
tw_grid, fhat = tweedie(x, grid)
xb = x.mean(); b = min(1.0, (p - 3) / np.sum((x - xb) ** 2))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
ax.hist(x, bins=40, density=True, color=PALETTE["light"], edgecolor="white", lw=0.5, label=r"observed $X_i$")
ax.plot(grid, marginal(grid), color=PALETTE["ink"], lw=1.4, label=r"true $f$")
ax.plot(grid, fhat, color=PALETTE["red"], lw=1.4, ls=(0, (4, 3)), label=r"fitted $\hat f$ (Poisson regression)")
ax.set_xlim(-3.5, 6.5)
ax.set_ylim(0, 0.36)
ax.set_xlabel(r"$x$")
ax.set_ylabel("density")
ax.set_title(r"Marginal density of $X$, $p = 1000$", loc="left")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.plot(grid, grid, color=PALETTE["grey"], lw=0.9, ls=":", label=r"MLE $\hat\theta_i = X_i$")
ax.plot(grid, xb + (1 - b) * (grid - xb), color=PALETTE["green"], lw=1.6, label="James–Stein (linear)")
inside = (grid >= np.quantile(x, 0.005)) & (grid <= np.quantile(x, 0.995))
ax.plot(grid[inside], tw_grid[inside], color=PALETTE["red"], lw=1.6, ls=(0, (4, 3)), label="Tweedie, estimated")
ax.plot(grid, oracle(grid), color=PALETTE["ink"], lw=1.4, label="oracle Bayes rule")
ax.set_xlim(-3.5, 6.5)
ax.set_ylim(-1.5, 5.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"estimate of $\theta$")
ax.set_title("Estimated posterior means", loc="left")
ax.legend(loc="upper left", fontsize=9)
save(fig, __file__, "eb-rules")

# risks over replications (printed for the text)
reps = 200
rng = np.random.default_rng(1957)
res = np.zeros((reps, 4))
for r in range(reps):
    th, xx = draw(rng)
    res[r] = [np.mean((xx - th) ** 2), np.mean((james_stein(xx) - th) ** 2),
              np.mean((tweedie(xx, xx)[0] - th) ** 2), np.mean((oracle(xx) - th) ** 2)]
print("MLE, JS, Tweedie, oracle:", res.mean(0).round(3), "B", round(b, 3))
