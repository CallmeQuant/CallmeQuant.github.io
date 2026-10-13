"""Quasi-likelihood with a misspecified variance function (note: quasi-likelihood and the variance function).
Gamma-type data with Var(Y) = phi * mu^2 and log mu = b0 + b1 x are fitted by quasi-score equations with
working variance V(mu) = mu^q. Left: asymptotic relative efficiency of the slope. Right: coverage of
model-based and sandwich intervals."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(1974)
BETA = np.array([0.5, 1.0]); PHI = 0.5; n = 200
x = np.linspace(0, 3, n)
X = np.column_stack([np.ones(n), x])
mu0 = np.exp(X @ BETA)


def asym_cov(q, mu=mu0):
    """Sandwich covariance (per observation scaling) of the quasi-score root with V(mu) = mu^q."""
    D = X * mu[:, None]                      # d mu / d beta for the log link
    A = D / mu[:, None] ** q                 # quasi-score weights D_i / V(mu_i)
    bread = np.linalg.inv(A.T @ D)
    meat = (A * (PHI * mu**2)[:, None]).T @ A
    return bread @ meat @ bread.T


def fit(y, q, iters=50):
    b = np.array([np.log(y.mean()), 0.0])
    for _ in range(iters):
        mu = np.exp(X @ b)
        D = X * mu[:, None]
        A = D / mu[:, None] ** q
        step = np.linalg.solve(A.T @ D, A.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(X @ b)
    D = X * mu[:, None]; A = D / mu[:, None] ** q
    bread = np.linalg.inv(A.T @ D)
    phi_hat = np.sum((y - mu) ** 2 / mu**q) / (n - 2)
    cov_model = phi_hat * bread                               # phi (D' V^-1 D)^-1
    cov_sand = bread @ ((A * ((y - mu) ** 2)[:, None]).T @ A) @ bread.T
    return b, cov_model, cov_sand


qs_fine = np.linspace(0, 3, 121)
v_opt = asym_cov(2.0)[1, 1]
eff = [v_opt / asym_cov(q)[1, 1] for q in qs_fine]

qs = np.arange(0, 3.01, 0.5)
R = 2000
shape = 1 / PHI
sim_eff, cov_m, cov_s = [], [], []
z = stats.norm.ppf(0.975)
Y = rng.gamma(shape, mu0 / shape, size=(R, n))   # E Y = mu0, Var Y = PHI * mu0^2
for q in qs:
    est, hm, hs = [], 0, 0
    for r in range(R):
        b, cm, cs = fit(Y[r], q)
        est.append(b[1])
        hm += abs(b[1] - BETA[1]) <= z * np.sqrt(cm[1, 1])
        hs += abs(b[1] - BETA[1]) <= z * np.sqrt(cs[1, 1])
    sim_eff.append(np.var(est))
    cov_m.append(hm / R); cov_s.append(hs / R)
sim_eff = np.array(sim_eff)
sim_eff = sim_eff[qs == 2.0][0] / sim_eff

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
ax = axes[0]
ax.plot(qs_fine, eff, color=PALETTE["blue"], label="asymptotic")
ax.plot(qs, sim_eff, "o", color=PALETTE["blue"], mfc="white", mew=1.3, label=rf"simulation, $n={n}$")
ax.axvline(2, color=PALETTE["grey"], lw=0.8, ls=":")
ax.set_xlabel(r"working variance power $q$ in $V(\mu) = \mu^q$")
ax.set_ylabel(r"efficiency of $\hat\beta_1$ relative to $q = 2$")
ax.set_ylim(0, 1.05)
ax.set_title("Efficiency", loc="left")
ax.legend(loc="lower right")

ax = axes[1]
ax.axhline(0.95, color=PALETTE["ink"], lw=0.8)
ax.plot(qs, cov_m, "-o", color=PALETTE["red"], mfc="white", mew=1.3, label=r"model-based: $\hat\phi\,(D^\top V^{-1} D)^{-1}$")
ax.plot(qs, cov_s, "-s", color=PALETTE["blue"], mfc="white", mew=1.3, label="sandwich")
ax.axvline(2, color=PALETTE["grey"], lw=0.8, ls=":")
ax.set_xlabel(r"working variance power $q$")
ax.set_ylabel(r"coverage of nominal 95% interval")
ax.set_ylim(0.7, 1.0)
ax.set_title(r"Confidence intervals for $\beta_1$", loc="left")
ax.legend(loc="lower center", fontsize=8.6)
save(fig, __file__, "working-variance")
print("eff", dict(zip(qs, np.round(sim_eff, 3))), "cov_m", np.round(cov_m, 3), "cov_s", np.round(cov_s, 3))
