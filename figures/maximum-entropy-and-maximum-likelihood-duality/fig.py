"""Duality between maximum entropy and maximum likelihood on X = {0,1,2,3}, T(x) = x."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(7)
x = np.arange(4)
true_p = np.array([0.15, 0.45, 0.10, 0.30])
n = 30
data = rng.choice(x, size=n, p=true_p)
p_hat = np.bincount(data, minlength=4) / n
t_hat = data.mean()

A = lambda th: np.log(np.exp(np.outer(np.atleast_1d(th), x)).sum(1))
mean_of = lambda th: (x * np.exp(th * x)).sum() / np.exp(th * x).sum()
pmf = lambda th: np.exp(th * x - A(th))
theta_hat = brentq(lambda th: mean_of(th) - t_hat, -30, 30)
H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()

fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2), gridspec_kw={"width_ratios": [1.05, 1.15, 0.9]})

# (a) dual objective g(theta) = A(theta) - theta * t_hat (minus average log-likelihood)
ax = axes[0]
th = np.linspace(-2.5, 2.0, 400)
ax.plot(th, A(th) - th * t_hat, color=PALETTE["blue"])
g_min = A(theta_hat)[0] - theta_hat * t_hat
ax.axhline(H(p_hat), color=PALETTE["red"], lw=1.1, ls="--")
ax.axhline(g_min, color=PALETTE["ink"], lw=0.7, ls=":")
ax.plot([theta_hat], [g_min], "o", color=PALETTE["ink"], ms=5)
ax.annotate(r"$\hat\theta$", (theta_hat, g_min), xytext=(4, -15), textcoords="offset points", fontsize=10)
ax.text(-2.4, H(p_hat) - 0.06, r"$H(\hat p_n)$", color=PALETTE["red"], va="top", fontsize=10)
ax.set_xlim(-2.5, 2.0); ax.set_ylim(0.9, 2.6)
ax.set_xlabel(r"$\theta$")
ax.set_title(r"Dual: $A(\theta) - \theta\,\hat t = -\ell_n(\theta)$", loc="left")

# (b) entropy of the maximum entropy pmf as the lower envelope of lines
ax = axes[1]
ms = np.linspace(0.02, 2.98, 300)
th_of_m = [brentq(lambda t: mean_of(t) - m, -40, 40) for m in ms]
Hstar = np.array([H(pmf(t)) for t in th_of_m])
for t0 in [-1.5, -0.6, 0.7, 1.6]:
    ax.plot(ms, A(t0)[0] - t0 * ms, color=PALETTE["grey"], lw=0.9)
ax.plot(ms, A(theta_hat)[0] - theta_hat * ms, color=PALETTE["ink"], lw=0.9, ls="--")
ax.plot(ms, Hstar, color=PALETTE["blue"], lw=2.0, label=r"$\max\{H(p) : \mathbb{E}_p X = m\}$")
ax.plot([t_hat], [H(pmf(theta_hat))], "o", color=PALETTE["ink"], ms=5)
ax.plot([t_hat], [H(p_hat)], "o", color=PALETTE["red"], ms=5)
ax.annotate(r"$\hat p_n$", (t_hat, H(p_hat)), xytext=(6, -4), textcoords="offset points",
            color=PALETTE["red"], fontsize=10)
ax.annotate(r"$p_{\hat\theta}$", (t_hat, H(pmf(theta_hat))), xytext=(-12, 11), textcoords="offset points", fontsize=10)
ax.set_xlim(0, 3); ax.set_ylim(0, 1.75)
ax.set_xlabel(r"mean $m$"); ax.set_ylabel("entropy (nats)")
ax.set_title(r"Primal: lines $m \mapsto A(\theta) - \theta m$", loc="left")
ax.legend(loc="lower center", fontsize=9)

# (c) empirical pmf versus fitted pmf (same mean)
ax = axes[2]
w = 0.38
ax.bar(x - w / 2, p_hat, width=w, color=PALETTE["red"], label=r"$\hat p_n$")
ax.bar(x + w / 2, pmf(theta_hat), width=w, color=PALETTE["blue"], label=r"$p_{\hat\theta}$")
ax.set_xticks(x); ax.set_ylim(0, 0.62)
ax.set_xlabel(r"$x$"); ax.set_ylabel("probability")
ax.set_title(rf"Same mean $\hat t = {t_hat:.2f}$", loc="left")
ax.legend(loc="upper left", ncol=2)
ax.grid(axis="x", visible=False)
save(fig, __file__, "duality")
print(t_hat, theta_hat, H(p_hat), H(pmf(theta_hat)), p_hat, pmf(theta_hat))
