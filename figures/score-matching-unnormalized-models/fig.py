"""Score matching for the unnormalized double-well family log q(x) = -t1 x^2/2 - t2 x^4/4,
compared with maximum likelihood (note: score matching)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from style import use_style, PALETTE, save

use_style()
theta0 = np.array([-2.0, 1.0])                  # log q = x^2 - x^4/4: modes at +-sqrt(2)
logq = lambda x, t: -t[0] * x**2 / 2 - t[1] * x**4 / 4
grid = np.linspace(-5, 5, 20001); dx = grid[1] - grid[0]


def sample(n, rng):
    """Exact draws by rejection from N(0, 1.5^2)."""
    s = 1.5
    logM = np.max(logq(grid, theta0) + grid**2 / (2 * s**2))          # log sup q/g, up to constants
    out = np.empty(0)
    while out.size < n:
        x = rng.normal(0, s, 2 * n)
        keep = np.log(rng.random(2 * n)) < logq(x, theta0) + x**2 / (2 * s**2) - logM
        out = np.concatenate([out, x[keep]])
    return out[:n]


def score_matching(x):
    """Closed form: minimize (1/2) t'A t + t'c with T'(x) = (-x, -x^3), T''(x) = (-1, -3x^2)."""
    A = np.array([[np.mean(x**2), np.mean(x**4)], [np.mean(x**4), np.mean(x**6)]])
    c = np.array([-1.0, -3 * np.mean(x**2)])
    return -np.linalg.solve(A, c)


def log_Z(t):
    v = logq(grid, t); m = v.max()
    return m + np.log(np.sum(np.exp(v - m)) * dx)


def mle(x, start):
    m2, m4 = np.mean(x**2), np.mean(x**4)
    nll = lambda t: t[0] * m2 / 2 + t[1] * m4 / 4 + log_Z(t) if t[1] > 0 else np.inf
    return minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10}).x


rng = np.random.default_rng(2005)
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})

# (a) one data set and the fitted densities
x = sample(1000, rng)
t_sm, t_ml = score_matching(x), None
t_ml = mle(x, t_sm)
xs = np.linspace(-3.2, 3.2, 600)
ax = axes[0]
ax.hist(x, bins=np.linspace(-3.2, 3.2, 50), density=True, color=PALETTE["light"], edgecolor="white", lw=0.4)
for t, col, lab, ls in [(theta0, PALETTE["ink"], r"true $\theta_0 = (-2, 1)$", "-"),
                        (t_sm, PALETTE["blue"], rf"score matching $({t_sm[0]:.2f}, {t_sm[1]:.2f})$", "-"),
                        (t_ml, PALETTE["red"], rf"maximum likelihood $({t_ml[0]:.2f}, {t_ml[1]:.2f})$", "--")]:
    ax.plot(xs, np.exp(logq(xs, t) - log_Z(t)), color=col, ls=ls, lw=1.4, label=lab)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 0.62)
ax.set_xlabel("$x$"); ax.set_ylabel("density")
ax.set_title(r"$n = 1000$ draws from $q(x) \propto e^{x^2 - x^4/4}$", loc="left")
ax.legend(loc="upper center", fontsize=8.6)

# (b) sampling distribution of the two estimators
R, n = 400, 500
est_sm, est_ml = np.empty((R, 2)), np.empty((R, 2))
for r in range(R):
    x = sample(n, rng)
    est_sm[r] = score_matching(x)
    est_ml[r] = mle(x, est_sm[r])
ax = axes[1]
ax.scatter(est_sm[:, 0], est_sm[:, 1], s=7, color=PALETTE["blue"], alpha=0.5, lw=0, label="score matching")
ax.scatter(est_ml[:, 0], est_ml[:, 1], s=7, color=PALETTE["red"], alpha=0.5, lw=0, label="maximum likelihood")
ang = np.linspace(0, 2 * np.pi, 200)
for est, col in [(est_sm, PALETTE["blue"]), (est_ml, PALETTE["red"])]:
    L = np.linalg.cholesky(np.cov(est.T))                  # 95% normal-theory ellipse
    ell = est.mean(0) + np.sqrt(5.991) * np.column_stack([np.cos(ang), np.sin(ang)]) @ L.T
    ax.plot(ell[:, 0], ell[:, 1], color=col, lw=1.4)
ax.plot(*theta0, "+", color=PALETTE["ink"], ms=12, mew=1.6)
ax.set_xlabel(r"$\hat\theta_1$"); ax.set_ylabel(r"$\hat\theta_2$")
sd_sm, sd_ml = est_sm.std(0), est_ml.std(0)
ax.set_title(rf"{R} data sets of size $n = {n}$", loc="left")
ax.legend(loc="lower left", fontsize=8.6, markerscale=2)
save(fig, __file__, "double-well")
print("sd SM", sd_sm, "sd ML", sd_ml, "ratio var", (sd_sm / sd_ml) ** 2)
