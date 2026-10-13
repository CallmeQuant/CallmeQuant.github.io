"""Huber location M-estimators: score functions and asymptotic variance as a function of the
threshold k, with Monte Carlo checks (note: M-estimators and the sandwich variance)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(20261012)

# Three symmetric error laws, all centred at 0.
EPS, TAU = 0.1, 5.0
laws = {
    "normal": (stats.norm(), lambda m: rng.standard_normal(m)),
    "contam": (None, lambda m: np.where(rng.random(m) < EPS, TAU, 1.0) * rng.standard_normal(m)),
    "t3": (stats.t(3), lambda m: rng.standard_t(3, m)),
}
labels = {"normal": r"$N(0,1)$",
          "contam": r"$0.9\,N(0,1) + 0.1\,N(0,25)$",
          "t3": r"Student $t_3$"}
colors = {"normal": PALETTE["blue"], "contam": PALETTE["red"], "t3": PALETTE["green"]}


def pdf(name, x):
    if name == "contam":
        return (1 - EPS) * stats.norm.pdf(x) + EPS * stats.norm.pdf(x, scale=TAU)
    return laws[name][0].pdf(x)


def cdf(name, x):
    if name == "contam":
        return (1 - EPS) * stats.norm.cdf(x) + EPS * stats.norm.cdf(x, scale=TAU)
    return laws[name][0].cdf(x)


def avar(name, k):
    """B / V^2 with B = E min(X^2, k^2), V = P(|X| < k)."""
    inner = integrate.quad(lambda x: x * x * pdf(name, x), 0, k)[0] * 2
    B = inner + k * k * 2 * (1 - cdf(name, k))
    V = 2 * cdf(name, k) - 1
    return B / V**2


def huber_location(X, k, iters=60):
    """Solve sum_i clip(X_i - theta, -k, k) = 0 row-wise by bisection (the map is monotone)."""
    lo, hi = X.min(axis=1), X.max(axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = np.clip(X - mid[:, None], -k, k).sum(axis=1)
        lo = np.where(s > 0, mid, lo)
        hi = np.where(s > 0, hi, mid)
    return 0.5 * (lo + hi)


ks = np.linspace(0.05, 4.0, 160)
k_sim = [0.25, 0.75, 1.345, 2.0, 3.0]
n, R = 200, 4000

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
u = np.linspace(-3, 3, 601)
ax.plot(u, u, color=PALETTE["grey"], label=r"mean: $\varphi(u)=u$")
ax.plot(u, np.clip(u, -1.345, 1.345), color=PALETTE["blue"], label=r"Huber, $k=1.345$")
ax.plot(u[u < 0], -np.ones((u < 0).sum()), color=PALETTE["gold"], label=r"median: $\mathrm{sign}(u)$")
ax.plot(u[u > 0], np.ones((u > 0).sum()), color=PALETTE["gold"])
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel(r"residual $u = x - \theta$")
ax.set_ylabel(r"$\varphi(u)$")
ax.set_title("Score functions", loc="left")
ax.legend(loc="upper left", fontsize=8.8)

ax = axes[1]
for name in laws:
    v = np.array([avar(name, k) for k in ks])
    ax.plot(ks, v, color=colors[name], label=labels[name])
    est = []
    for k in k_sim:
        X = laws[name][1](R * n).reshape(R, n)
        est.append(n * huber_location(X, k).var())
    ax.plot(k_sim, est, "o", color=colors[name], ms=4.5, mfc="white", mew=1.3)
ax.axvline(1.345, color=PALETTE["grey"], lw=0.8, ls=":")
ax.set_xlim(0, 4.05)
ax.set_ylim(0.9, 3.5)
ax.set_xlabel(r"threshold $k$")
ax.set_ylabel(r"$\sigma^2(k) = B/V^{2}$")
ax.set_title(r"Asymptotic variance of $\sqrt{n}(\hat\theta_n-\theta_0)$", loc="left")
ax.legend(loc="upper right", fontsize=8.8)
save(fig, __file__, "huber-variance")

for name in laws:
    print(name, "k->0 (median):", round(avar(name, 1e-4), 3), " k=1.345:", round(avar(name, 1.345), 3),
          " k=4:", round(avar(name, 4.0), 3))
