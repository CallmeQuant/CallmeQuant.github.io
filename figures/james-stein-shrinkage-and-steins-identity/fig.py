"""James-Stein estimation (note: shrinking unrelated means).
Figure 1: exact risk of the James-Stein estimator relative to the MLE, as a function of the norm of theta.
Figure 2: one draw with p = 30; MLE and James-Stein estimates against the true means."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1961)


def js_risk(p, lam):
    """R(theta) = p - (p-2)^2 E[1/(p-2+2K)], K ~ Poisson(lam/2), lam = ||theta||^2."""
    k = np.arange(0, 2000)
    w = stats.poisson.pmf(k, lam / 2)
    return p - (p - 2) ** 2 * np.sum(w / (p - 2 + 2 * k))


# ---- Figure 1 ----
r = np.linspace(0, 12, 241)
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for col, p in zip(CYCLE, [3, 5, 10, 30]):
    ax.plot(r, [js_risk(p, t * t) / p for t in r], color=col, label=rf"$p = {p}$")
# positive-part James-Stein for p = 10, by simulation
p, R = 10, 40000
rs = np.arange(0, 12.1, 1.0)
pp = []
for t in rs:
    theta = np.zeros(p); theta[0] = t
    X = theta + rng.standard_normal((R, p))
    shrink = np.maximum(0, 1 - (p - 2) / np.sum(X**2, axis=1))
    pp.append(np.mean(np.sum((shrink[:, None] * X - theta) ** 2, axis=1)) / p)
ax.plot(rs, pp, "o", color=CYCLE[2], mfc="white", mew=1.2, ms=4.5, label=r"positive part, $p = 10$ (simulated)")
ax.axhline(1, color=PALETTE["ink"], lw=0.8, ls="--")
ax.text(11.9, 1.015, "MLE", ha="right", va="bottom", fontsize=9.5)
ax.set_xlim(0, 12); ax.set_ylim(0, 1.08)
ax.set_xlabel(r"$\Vert \theta \Vert$")
ax.set_ylabel(r"risk relative to the MLE, $R(\theta)/p$")
ax.legend(loc="lower right", fontsize=9)
save(fig, __file__, "js-risk")

# ---- Figure 2 ----
p = 30
theta = 2.0 + rng.normal(0, 0.8, p)                      # means clustered away from 0
X = theta + rng.standard_normal(p)
js0 = (1 - (p - 2) / np.sum(X**2)) * X
xbar = X.mean()
jsm = xbar + (1 - (p - 3) / np.sum((X - xbar) ** 2)) * (X - xbar)
err = lambda e: np.sum((e - theta) ** 2)
fig, ax = plt.subplots(figsize=(6.6, 3.7))
lo, hi = min(theta.min(), X.min()) - 0.3, max(theta.max(), X.max()) + 0.3
ax.plot([lo, hi], [lo, hi], color=PALETTE["grey"], lw=0.8)
for a, b, c in zip(theta, X, jsm):
    ax.plot([a, a], [b, c], color=PALETTE["light"], lw=1.0, zorder=1)
ax.plot(theta, X, "o", color=PALETTE["grey"], ms=4.5, label=rf"MLE $X_i$: loss {err(X):.1f}")
ax.plot(theta, js0, "^", color=PALETTE["gold"], ms=4.5, mfc="white", mew=1.1, label=rf"James–Stein toward 0: loss {err(js0):.1f}")
ax.plot(theta, jsm, "s", color=PALETTE["blue"], ms=4.2, label=rf"James–Stein toward $\bar X$: loss {err(jsm):.1f}")
ax.axhline(xbar, color=PALETTE["blue"], lw=0.7, ls=":")
ax.set_xlim(theta.min() - 0.3, theta.max() + 0.3); ax.set_ylim(lo, hi)
ax.set_xlabel(r"true mean $\theta_i$"); ax.set_ylabel("estimate")
ax.legend(loc="upper left", fontsize=8.8)
ax.set_title(rf"One draw, $p = {p}$; dotted line: $\bar X$", loc="left")
save(fig, __file__, "js-one-draw")
print(err(X), err(js0), err(jsm), np.sum(theta**2))
