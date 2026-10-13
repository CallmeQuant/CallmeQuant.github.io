"""One posterior, four computations: a two-parameter logistic regression (n = 25) computed exactly on
a grid, by the Laplace approximation, by random-walk Metropolis and by mean-field Gaussian
variational inference."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit, log_expit
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(1986)
n, prior_sd = 25, 2.5
x = rng.uniform(0.0, 3.0, n)
y = (rng.random(n) < expit(-2.0 + 1.5 * x)).astype(float)


def loglik(a, b):
    eta = a[..., None] + b[..., None] * x
    return np.sum(y * log_expit(eta) + (1 - y) * log_expit(-eta), axis=-1)


def logpost(a, b):
    return loglik(a, b) - (a**2 + b**2) / (2 * prior_sd**2)


def neg(f):
    return lambda t: -f(np.array(t[0]), np.array(t[1]))


# point estimates
mle = minimize(neg(loglik), [0.0, 0.0], method="BFGS").x
mode = minimize(neg(logpost), [0.0, 0.0], method="BFGS").x


def hessian(f, t, h=1e-4):
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (f(*(t + ei + ej)) - f(*(t + ei - ej)) - f(*(t - ei + ej)) + f(*(t - ei - ej))) / (4 * h * h)
    return H


lp2 = lambda a, b: float(logpost(np.array(a), np.array(b)))
lap_cov = np.linalg.inv(-hessian(lp2, mode))

# exact posterior on a grid
A = np.linspace(-5.5, 3.0, 501); B = np.linspace(-1.5, 3.5, 501)
AA, BB = np.meshgrid(A, B)
LP = logpost(AA, BB)
P = np.exp(LP - LP.max()); dA, dB = A[1] - A[0], B[1] - B[0]
P /= P.sum() * dA * dB
post_mean = np.array([(AA * P).sum(), (BB * P).sum()]) * dA * dB

# random-walk Metropolis with proposal 2.38^2/d times the Laplace covariance
iters, burn = 20000, 2000
L = np.linalg.cholesky(2.38**2 / 2 * lap_cov)
cur = mode.copy(); cur_lp = lp2(*cur); chain = np.empty((iters, 2)); acc = 0
for t in range(iters):
    prop = cur + L @ rng.standard_normal(2)
    prop_lp = lp2(*prop)
    if np.log(rng.random()) < prop_lp - cur_lp:
        cur, cur_lp = prop, prop_lp; acc += 1
    chain[t] = cur
chain = chain[burn:]

# mean-field Gaussian VI: maximize the ELBO, expectations by 30 x 30 Gauss-Hermite quadrature
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(30)
gh_w = gh_w / gh_w.sum()


def neg_elbo(par):
    m1, m2, ls1, ls2 = par
    a = m1 + np.exp(ls1) * gh_x[:, None]; b = m2 + np.exp(ls2) * gh_x[None, :]
    e = np.sum(gh_w[:, None] * gh_w[None, :] * logpost(a, b))
    return -(e + ls1 + ls2)      # entropy up to a constant


vi = minimize(neg_elbo, [mode[0], mode[1], 0.0, 0.0], method="Nelder-Mead",
              options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000}).x
vi_m, vi_s = vi[:2], np.exp(vi[2:])

print("MLE", mle.round(3), "MAP", mode.round(3), "post mean", post_mean.round(3))
print("exact sd", np.sqrt(np.array([((AA - post_mean[0])**2 * P).sum(), ((BB - post_mean[1])**2 * P).sum()]) * dA * dB).round(3))
print("laplace mode", mode, "laplace sd", np.sqrt(np.diag(lap_cov)).round(3), "corr", (lap_cov[0, 1] / np.sqrt(lap_cov[0, 0] * lap_cov[1, 1])).round(3))
print("mcmc mean", chain.mean(0).round(3), "sd", chain.std(0).round(3), "acc", acc / iters)
print("vi mean", vi_m.round(3), "sd", vi_s.round(3))
print("sum y", y.sum())


def gauss_density(m, C):
    Ci = np.linalg.inv(C); d = np.stack([AA - m[0], BB - m[1]], -1)
    q = np.einsum("...i,ij,...j->...", d, Ci, d)
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(np.linalg.det(C)))


def levels(D):
    """Contour levels enclosing 50%, 90% and 99% of the mass of a gridded density."""
    s = np.sort(D.ravel())[::-1]; c = np.cumsum(s) * dA * dB
    return sorted(s[np.searchsorted(c, q)] for q in (0.99, 0.9, 0.5))


fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
lev = levels(P)
panels = [("Exact (grid)", None), ("Laplace approximation", gauss_density(mode, lap_cov)),
          ("Random-walk Metropolis", None), ("Mean-field variational", gauss_density(vi_m, np.diag(vi_s**2)))]
for k, (ax, (title, Q)) in enumerate(zip(axes.ravel(), panels)):
    ax.contour(AA, BB, P, levels=lev, colors=PALETTE["grey"] if k else PALETTE["ink"], linewidths=1.0)
    if Q is not None:
        ax.contour(AA, BB, Q, levels=levels(Q), colors=CYCLE[k], linewidths=1.3, linestyles="dashed")
    if k == 2:
        idx = rng.choice(len(chain), 1500, replace=False)
        ax.scatter(chain[idx, 0], chain[idx, 1], s=2.5, color=PALETTE["green"], alpha=0.5, lw=0)
    if k == 0:
        ax.plot(*mle, "s", color=PALETTE["red"], ms=5, label="MLE")
        ax.plot(*mode, "o", color=PALETTE["blue"], ms=5, label="MAP")
        ax.plot(*post_mean, "^", color=PALETTE["gold"], ms=6, label="posterior mean")
        ax.legend(loc="lower left", fontsize=8.8)
    ax.set_title(title, loc="left")
    ax.set_xlim(-4.2, 1.8); ax.set_ylim(-1.0, 2.6)
for ax in axes[1]:
    ax.set_xlabel(r"intercept $\alpha$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"slope $\beta$")
save(fig, __file__, "four-posteriors", fmt="png")

# marginal posterior of the slope
fig, ax = plt.subplots(figsize=(6.4, 3.2))
pb = P.sum(1) * dA
ax.hist(chain[:, 1], bins=70, range=(-0.6, 2.6), density=True, color=PALETTE["light"], edgecolor="white", lw=0.4,
        label="Metropolis samples")
ax.plot(B, pb, color=PALETTE["ink"], lw=1.8, label="exact")
ax.plot(B, norm.pdf(B, mode[1], np.sqrt(lap_cov[1, 1])), color=PALETTE["red"], lw=1.6, ls=(0, (4, 3)), label="Laplace")
ax.plot(B, norm.pdf(B, vi_m[1], vi_s[1]), color=PALETTE["violet"], lw=1.6, ls=(0, (1, 1.5)), label="mean-field VI")
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, 1.15 * max(pb.max(), norm.pdf(0, 0, vi_s[1])))
ax.set_xlabel(r"slope $\beta$")
ax.set_ylabel("marginal posterior density")
ax.legend(loc="upper right")
save(fig, __file__, "slope-marginal")
