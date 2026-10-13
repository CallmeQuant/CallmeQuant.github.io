"""Gaussian fits to a two-component mixture under reverse and forward KL, and mean-field fits to a
correlated Gaussian (note: zero-forcing in reverse Kullback-Leibler fits)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()
# Expectations under N(0, 1) by the trapezoidal rule on a fine grid. (Gauss-Hermite rules converge
# slowly here because log p_a has a sharp bend near x = 0, which shows up as spurious wiggles.)
GH_X = np.linspace(-10, 10, 4001)
GH_W = norm.pdf(GH_X)
GH_W = GH_W / GH_W.sum()


def log_p(x, a, w):
    """log density of w N(a, 1) + (1 - w) N(-a, 1)."""
    return np.logaddexp(np.log(w) - 0.5 * (x - a) ** 2,
                        np.log1p(-w) - 0.5 * (x + a) ** 2) - 0.5 * np.log(2 * np.pi)


def reverse_kl(m, s, a, w):
    """KL(N(m, s^2) || p) by quadrature in the standardized variable."""
    x = m + s * GH_X
    return np.sum(GH_W * (norm.logpdf(x, m, s) - log_p(x, a, w)))


def reverse_profile(m, a, w):
    """min over s of KL(N(m, s^2) || p): coarse grid in log s, then a bounded refinement."""
    ls = np.linspace(np.log(0.2), np.log(8.0), 120)
    vals = [reverse_kl(m, np.exp(l), a, w) for l in ls]
    k = int(np.argmin(vals))
    lo, hi = ls[max(k - 1, 0)], ls[min(k + 1, len(ls) - 1)]
    r = minimize_scalar(lambda l: reverse_kl(m, np.exp(l), a, w), bounds=(lo, hi), method="bounded")
    return r.fun, np.exp(r.x)


def mixture_entropy(a, w):
    x = np.linspace(-a - 12, a + 12, 20001)
    lp = log_p(x, a, w)
    return -np.trapezoid(np.exp(lp) * lp, x)


def forward_profile(m, a, w):
    """min over s of KL(p || N(m, s^2)) = -H(p) + log(2 pi e (sigma_p^2 + (m - mu_p)^2)) / 2."""
    mu, var = (2 * w - 1) * a, 1 + 4 * w * (1 - w) * a ** 2
    return -mixture_entropy(a, w) + 0.5 * np.log(2 * np.pi * np.e * (var + (m - mu) ** 2))


# ---------- Figure 1: unequal weights, well separated --------------------------------------
a, w = 3.2, 0.6
mgrid = np.linspace(-6, 6, 241)
rev = np.array([reverse_profile(m, a, w) for m in mgrid])
fwd = np.array([forward_profile(m, a, w) for m in mgrid])
# local minima of the reverse profile, refined
loc = [i for i in range(1, len(mgrid) - 1) if rev[i, 0] < rev[i - 1, 0] and rev[i, 0] < rev[i + 1, 0]]
fits = []
for i in loc:
    r = minimize_scalar(lambda m: reverse_profile(m, a, w)[0], bounds=(mgrid[i - 1], mgrid[i + 1]),
                        method="bounded")
    fits.append((r.x, reverse_profile(r.x, a, w)[1], r.fun))
fits.sort(key=lambda t: t[2])
print("reverse-KL local minima (m, s, KL):", [tuple(np.round(f, 3)) for f in fits])
mu_p, sd_p = (2 * w - 1) * a, np.sqrt(1 + 4 * w * (1 - w) * a ** 2)

x = np.linspace(-7.5, 7.5, 800)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axes[0]
ax.fill_between(x, np.exp(log_p(x, a, w)), color=PALETTE["light"], lw=0)
ax.plot(x, np.exp(log_p(x, a, w)), color=PALETTE["ink"], lw=1.4, label=r"target $p$")
m0, s0, _ = fits[0]
ax.plot(x, norm.pdf(x, m0, s0), color=PALETTE["red"], label=r"$\arg\min\ \mathrm{KL}(q\,\Vert\,p)$")
if len(fits) > 1:
    m1, s1, _ = fits[1]
    ax.plot(x, norm.pdf(x, m1, s1), color=PALETTE["red"], ls="--", lw=1.3, label="its other local minimum")
ax.plot(x, norm.pdf(x, mu_p, sd_p), color=PALETTE["blue"], label=r"$\arg\min\ \mathrm{KL}(p\,\Vert\,q)$")
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(0, 0.72)
ax.set_xlabel(r"$x$")
ax.set_ylabel("density")
ax.set_title("Target and Gaussian fits")
ax.legend(loc="upper left", fontsize=8.6)

ax = axes[1]
ax.plot(mgrid, rev[:, 0], color=PALETTE["red"], label=r"$\min_s \mathrm{KL}(\mathcal{N}(m,s^2)\,\Vert\,p)$")
ax.plot(mgrid, fwd, color=PALETTE["blue"], label=r"$\min_s \mathrm{KL}(p\,\Vert\,\mathcal{N}(m,s^2))$")
ax.axhline(-np.log(w), color=PALETTE["grey"], lw=0.9, ls=":")
ax.axhline(-np.log(1 - w), color=PALETTE["grey"], lw=0.9, ls=":")
ax.text(-5.9, -np.log(w) - 0.04, r"$-\log w$", va="top", fontsize=9, color=PALETTE["grey"])
ax.text(-5.9, -np.log(1 - w) - 0.04, r"$-\log(1-w)$", va="top", fontsize=9, color=PALETTE["grey"])
ax.set_xlim(-6, 6)
ax.set_ylim(0, 3.0)
ax.set_xlabel(r"location $m$ of $q=\mathcal{N}(m,s^2)$")
ax.set_ylabel("divergence (profiled over $s$)")
ax.set_title("Profiles in the location")
ax.legend(loc="upper center", fontsize=8.6)
save(fig, __file__, "mixture-fits")

# ---------- Figure 2: equal weights, profiles for several separations ------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.3))
mgrid2 = np.linspace(-5, 5, 201)
for k, aa in enumerate([1.5, 2.5, 2.8, 3.5]):
    prof = np.array([reverse_profile(m, aa, 0.5)[0] for m in mgrid2])
    ax.plot(mgrid2, prof, color=CYCLE[k], label=rf"$a={aa}$")
ax.axhline(np.log(2), color=PALETTE["grey"], lw=0.9, ls=":")
ax.text(4.95, np.log(2) + 0.03, r"$\log 2$", ha="right", va="bottom", fontsize=9, color=PALETTE["grey"])
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1.8)
ax.set_xlabel(r"location $m$")
ax.set_ylabel(r"$\min_s \mathrm{KL}(\mathcal{N}(m,s^2)\,\Vert\,p_a)$")
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13))
save(fig, __file__, "separation-profiles")

# crossover separation (equal weights): where the central fit stops being the global minimum
def gap(aa):
    r = minimize_scalar(lambda m: reverse_profile(m, aa, 0.5)[0], bounds=(aa - 0.6, aa + 0.3), method="bounded")
    return reverse_profile(0.0, aa, 0.5)[0] - r.fun
lo, hi = 2.5, 3.1
for _ in range(40):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
print("crossover a (central vs component-centred local minimum):", round(0.5 * (lo + hi), 3))

# ---------- Figure 3: mean-field fits to a correlated bivariate Gaussian ---------------------
def ellipse(cov, r=2.0, n=300):
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(cov)
    return (L @ np.vstack([np.cos(t), np.sin(t)]) * r)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
for ax, rho in zip(axes, [0.6, 0.95]):
    S = np.array([[1.0, rho], [rho, 1.0]])
    Lam = np.linalg.inv(S)
    for cov, col, lab, ls in [(S, PALETTE["ink"], r"target $\mathcal{N}(0,\Sigma)$", "-"),
                              (np.diag(1 / np.diag(Lam)), PALETTE["red"], r"reverse KL: $v_i = 1/\Lambda_{ii}$", "-"),
                              (np.diag(np.diag(S)), PALETTE["blue"], r"forward KL: $v_i = \Sigma_{ii}$", "--")]:
        e = ellipse(cov)
        ax.plot(e[0], e[1], color=col, ls=ls, label=lab, lw=1.6 if col != PALETTE["ink"] else 1.3)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel(r"$x_1$")
    ax.set_title(rf"$\rho = {rho}$")
axes[0].set_ylabel(r"$x_2$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=9)
save(fig, __file__, "mean-field-ellipses")
