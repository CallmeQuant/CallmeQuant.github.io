"""Control variates for E[log(1 + exp(Z))], Z ~ N(0.5, 1.5^2) (note: control variates)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
mu, s = 0.5, 1.5
f = lambda z: np.logaddexp(0.0, z)                         # softplus

# Reference value by Gauss-Hermite quadrature (probabilists' weights).
x, w = np.polynomial.hermite_e.hermegauss(120)
theta = np.sum(w / w.sum() * f(mu + s * x))
var_f = np.sum(w / w.sum() * (f(mu + s * x) - theta) ** 2)


def controls(z, k):
    g = [z - mu, (z - mu) ** 2 - s**2]                      # both have mean zero
    return np.stack(g[:k], 1)


def estimate(z, k):
    """Control-variate estimate with k controls and OLS coefficient; k = 0 is the plain mean."""
    y = f(z)
    if k == 0:
        return y.mean()
    X = np.column_stack([np.ones_like(z), controls(z, k)])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef[0]                                          # fitted value where all controls equal their means


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})

# (a) the regression picture for one sample
rng = np.random.default_rng(5)
z = rng.normal(mu, s, 60)
g, y = z - mu, f(z)
b, a = np.polyfit(g, y, 1)
gg = np.linspace(-4, 4, 100)
ax = axes[0]
ax.scatter(g, y, s=12, color=PALETTE["grey"], alpha=0.8, lw=0)
ax.plot(gg, a + b * gg, color=PALETTE["blue"], lw=1.4)
ax.axhline(theta, color=PALETTE["ink"], lw=0.7, ls=":")
ax.axvline(0, color=PALETTE["ink"], lw=0.5)
ax.plot(g.mean(), y.mean(), "o", color=PALETTE["red"], ms=6, label=rf"plain mean $\bar f = {y.mean():.3f}$")
ax.plot(0, a, "s", color=PALETTE["blue"], ms=6, label=rf"line at $g = 0$: ${a:.3f}$")
ax.annotate(rf"$\theta = {theta:.3f}$", (3.9, theta), xytext=(3.9, theta - 0.55), ha="right", fontsize=9.5)
ax.set_xlim(-4, 4); ax.set_ylim(-0.3, 4.6)
ax.set_xlabel(r"control $g = Z - \mu$"); ax.set_ylabel(r"$f(Z) = \log(1 + e^{Z})$")
ax.set_title(r"One sample, $n = 60$", loc="left")
ax.legend(loc="upper left")

# (b) root mean squared error against n
ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
R = 2000
rng = np.random.default_rng(6)
labels = ["plain Monte Carlo", r"control $Z - \mu$", r"controls $Z - \mu$, $(Z-\mu)^2 - \sigma^2$"]
cols = [PALETTE["red"], PALETTE["blue"], PALETTE["green"]]
xg, wg = x, w / w.sum()
G = controls(mu + s * xg, 2); F = f(mu + s * xg) - theta
ax = axes[1]
for k in range(3):
    rmse = []
    for n in ns:
        est = np.array([estimate(rng.normal(mu, s, n), k) for _ in range(R)])
        rmse.append(np.sqrt(np.mean((est - theta) ** 2)))
    if k == 0:
        resid_var = var_f
    else:
        Sgg = (wg[:, None, None] * G[:, :k, None] * G[:, None, :k]).sum(0)
        Sgf = (wg[:, None] * G[:, :k] * F[:, None]).sum(0)
        resid_var = var_f - Sgf @ np.linalg.solve(Sgg, Sgf)
    nn = np.geomspace(8, 7000, 50)
    ax.plot(nn, np.sqrt(resid_var / nn), color=cols[k], lw=1.2, label=labels[k])
    ax.plot(ns, rmse, "o", color=cols[k], ms=4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sample size $n$"); ax.set_ylabel("root mean squared error")
ax.set_title(r"Error against $n$ (lines: $\sqrt{\sigma_f^2 (1 - R^2)/n}$)", loc="left")
ax.set_ylim(3e-4, 1.2)
ax.legend(loc="lower left", fontsize=8.8)
save(fig, __file__, "control-variates")
print(theta, var_f)
