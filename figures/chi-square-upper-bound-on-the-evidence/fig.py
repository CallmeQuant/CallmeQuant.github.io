"""ELBO and chi-square upper bound (CUBO_2) for a skew-normal posterior: optimal Gaussian fits, the
evidence sandwich, and Monte Carlo behaviour of the CUBO estimate (note: the chi-square upper bound)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm
from style import use_style, PALETTE, save

use_style()
ALPHA = 3.0                     # probit slope: prior z ~ N(0, 1), y = 1 with P(y = 1 | z) = Phi(ALPHA z)
LOG_EV = np.log(0.5)            # p(y = 1) = 1/2 by symmetry


def log_joint(z):
    return norm.logcdf(ALPHA * z) + norm.logpdf(z)


def elbo(m, s):
    f = lambda t: norm.pdf(t, m, s) * (log_joint(t) - norm.logpdf(t, m, s))
    return quad(f, m - 12 * s, m + 12 * s, limit=200)[0]


def cubo2(m, s):
    """(1/2) log E_q[(p(y, z) / q(z))^2]; infinite when s^2 < 1/2 (right tail of the posterior)."""
    if s * s < 0.5 or (s * s == 0.5 and m <= 0):
        return np.inf
    f = lambda t: np.exp(2 * log_joint(t) - norm.logpdf(t, m, s) - 2 * LOG_EV)
    v = quad(f, -np.inf, m, limit=400)[0] + quad(f, m, np.inf, limit=400)[0]
    return LOG_EV + 0.5 * np.log(v)


r = minimize(lambda v: -elbo(v[0], np.exp(v[1])), [0.75, np.log(0.6)], method="Nelder-Mead",
             options=dict(xatol=1e-7, fatol=1e-12))
m_e, s_e = r.x[0], np.exp(r.x[1])
r = minimize(lambda v: cubo2(v[0], np.exp(v[1])), [0.75, np.log(0.75)], method="Nelder-Mead",
             options=dict(xatol=1e-7, fatol=1e-12))
m_c, s_c = r.x[0], np.exp(r.x[1])
print(f"ELBO-optimal q: m={m_e:.4f}, s^2={s_e**2:.4f}, ELBO={elbo(m_e, s_e):.4f}")
print(f"CUBO-optimal q: m={m_c:.4f}, s^2={s_c**2:.4f}, CUBO={cubo2(m_c, s_c):.4f}, ELBO there={elbo(m_c, s_c):.4f}")
print("chi2 at CUBO optimum:", np.exp(2 * (cubo2(m_c, s_c) - LOG_EV)) - 1, " log p(y) =", LOG_EV)

# ---------------- Figure 1 ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax = axes[0]
z = np.linspace(-2.2, 3.4, 600)
post = np.exp(log_joint(z) - LOG_EV)
ax.fill_between(z, post, color=PALETTE["light"], lw=0)
ax.plot(z, post, color=PALETTE["ink"], lw=1.4, label=r"posterior $\pi$")
ax.plot(z, norm.pdf(z, m_e, s_e), color=PALETTE["blue"], label=r"$\arg\max$ ELBO")
ax.plot(z, norm.pdf(z, m_c, s_c), color=PALETTE["red"], label=r"$\arg\min\ \mathrm{CUBO}_2$")
ax.set_xlim(-2.2, 3.4)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$z$")
ax.set_ylabel("density")
ax.set_title("Posterior and Gaussian fits")
ax.legend(loc="upper left")

ax = axes[1]
s_grid = np.linspace(0.45, 1.3, 70)
el = [-minimize_scalar(lambda m: -elbo(m, s), bounds=(0.2, 1.3), method="bounded").fun for s in s_grid]
s_grid_c = np.concatenate([[np.sqrt(0.5)], np.linspace(np.sqrt(0.5) + 1e-4, 1.3, 70)])
cu = [minimize_scalar(lambda m: cubo2(m, s), bounds=(0.05, 1.5), method="bounded").fun for s in s_grid_c]
ax.axvspan(0.45, np.sqrt(0.5), color=PALETTE["light"], lw=0)
ax.text(0.465, -0.60, r"$\mathrm{CUBO}_2 = +\infty$" + "\n" + r"for $s^2 < 1/2$", fontsize=9,
        color=PALETTE["ink"], va="top")
ax.axhline(LOG_EV, color=PALETTE["ink"], lw=1.0, ls="--")
ax.text(1.29, LOG_EV + 0.004, r"$\log p(y)$", ha="right", va="bottom", fontsize=9.5)
ax.plot(s_grid, el, color=PALETTE["blue"], label=r"$\max_m \mathrm{ELBO}$")
ax.plot(s_grid_c, cu, color=PALETTE["red"], label=r"$\min_m \mathrm{CUBO}_2$")
ax.plot([s_e], [elbo(m_e, s_e)], "o", color=PALETTE["blue"], ms=4.5)
ax.plot([s_c], [cubo2(m_c, s_c)], "o", color=PALETTE["red"], ms=4.5)
ax.set_xlim(0.45, 1.3)
ax.set_ylim(-0.95, -0.56)
ax.set_xlabel(r"scale $s$ of $q=\mathcal{N}(m,s^2)$")
ax.set_ylabel("nats")
ax.set_title("The evidence sandwich")
ax.legend(loc="lower right")
save(fig, __file__, "fits-and-sandwich")

# ---------------- Figure 2: Monte Carlo CUBO estimates -------------------------------------
rng = np.random.default_rng(2017)
S_list = np.array([10, 30, 100, 300, 1000, 3000, 10000])
reps = 2000


def cubo_hat(m, s, S):
    out = np.empty(reps)
    for k in range(reps):
        zz = rng.normal(m, s, S)
        lw = log_joint(zz) - norm.logpdf(zz, m, s)
        out[k] = 0.5 * (logsumexp(2 * lw) - np.log(S))
    return out


fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
for ax, (m, s, col, title, truth) in zip(axes, [
        (m_c, s_c, PALETTE["red"], rf"$q = \arg\min\ \mathrm{{CUBO}}_2$ ($s^2 = {s_c**2:.3f}$)", cubo2(m_c, s_c)),
        (m_e, s_e, PALETTE["blue"], rf"$q = \arg\max$ ELBO ($s^2 = {s_e**2:.3f}$)", np.inf)]):
    q = np.array([np.quantile(cubo_hat(m, s, S), [0.1, 0.5, 0.9]) for S in S_list])
    print(np.round(q, 4))
    ax.fill_between(S_list, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0, label="10%–90% range")
    ax.plot(S_list, q[:, 1], color=col, marker="o", ms=3.5, label="median")
    ax.axhline(LOG_EV, color=PALETTE["ink"], lw=1.0, ls="--", label=r"$\log p(y)$")
    if np.isfinite(truth):
        ax.axhline(truth, color=col, lw=1.0, ls=":", label=r"true $\mathrm{CUBO}_2(q)$")
    ax.set_xscale("log")
    ax.set_xlabel(r"number of draws $S$")
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.6)
axes[0].set_ylabel(r"estimate of $\mathrm{CUBO}_2(q)$")
axes[0].set_ylim(-0.86, -0.42)
save(fig, __file__, "monte-carlo-cubo")
