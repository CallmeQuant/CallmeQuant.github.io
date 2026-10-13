"""Bayesian model averaging over all 2^6 linear regressions with Zellner's g-prior (g = n):
posterior inclusion probabilities, the model-averaged posterior of one coefficient, and
out-of-sample log predictive scores of BMA against single models (note: Bayesian model averaging)."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from style import use_style, PALETTE, CYCLE, save

use_style()
n, p = 60, 6
beta = np.array([0.6, 0.0, 0.4, 0.0, 0.0, 0.25])
R = np.full((p, p), 0.3)
np.fill_diagonal(R, 1.0)
R[0, 1] = R[1, 0] = 0.8
Lc = np.linalg.cholesky(R)
MODELS = [np.array(m, dtype=bool) for m in itertools.product([0, 1], repeat=p)]


def simulate(rng, m):
    X = rng.standard_normal((m, p)) @ Lc.T
    return X, X @ beta + rng.standard_normal(m)


def fit(X, y, g):
    """Per-model posterior summaries under p(alpha, sigma^2) ~ 1/sigma^2, beta ~ N(0, g sigma^2 (Xc'Xc)^-1)."""
    m = len(y)
    xbar, ybar = X.mean(0), y.mean()
    Xc, yc = X - xbar, y - ybar
    tss = yc @ yc
    out = []
    for gam in MODELS:
        k = gam.sum()
        if k:
            Xk = Xc[:, gam]
            XtX = Xk.T @ Xk
            bhat = np.linalg.solve(XtX, Xk.T @ yc)
            R2 = (yc @ Xk @ bhat) / tss
            XtXinv = np.linalg.inv(XtX)
        else:
            bhat, R2, XtXinv = np.zeros(0), 0.0, np.zeros((0, 0))
        logml = 0.5 * (m - 1 - k) * np.log1p(g) - 0.5 * (m - 1) * np.log1p(g * (1 - R2))
        S = tss * (1 - g / (1 + g) * R2)
        out.append(dict(gam=gam, logml=logml, mean=g / (1 + g) * bhat, S=S,
                        XtXinv=XtXinv, shrink=g / (1 + g)))
    lw = np.array([o["logml"] for o in out])
    w = np.exp(lw - lw.max())
    return out, w / w.sum(), xbar, ybar


def log_pred(out, w, xbar, ybar, Xnew, ynew, m):
    """Log predictive densities (Student t, m - 1 df) of new responses, for every model."""
    dens = []
    for o in out:
        Xc = (Xnew - xbar)[:, o["gam"]]
        loc = ybar + Xc @ o["mean"]
        quad = np.einsum("ij,jk,ik->i", Xc, o["XtXinv"], Xc) if o["gam"].any() else 0.0
        scale = np.sqrt(o["S"] / (m - 1) * (1 + 1 / m + o["shrink"] * quad))
        dens.append(tdist.logpdf(ynew, m - 1, loc, scale))
    return np.array(dens)  # models x points


# ---- one data set: inclusion probabilities and the posterior of beta_6
rng = np.random.default_rng(7)
X, y = simulate(rng, n)
out, w, xbar, ybar = fit(X, y, g=n)
G = np.array(MODELS)
pip = G.T @ w

grid = np.linspace(-0.3, 0.8, 800)
j = 5
cont = np.zeros_like(grid)
for o, wk in zip(out, w):
    if o["gam"][j]:
        pos = np.flatnonzero(o["gam"]).tolist().index(j)
        sd = np.sqrt(o["S"] / (n - 1) * o["shrink"] * o["XtXinv"][pos, pos])
        cont += wk * tdist.pdf(grid, n - 1, o["mean"][pos], sd)
full = out[-1]
sd_full = np.sqrt(full["S"] / (n - 1) * full["shrink"] * full["XtXinv"][j, j])
dens_full = tdist.pdf(grid, n - 1, full["mean"][j], sd_full)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
cols = [PALETTE["blue"] if b != 0 else PALETTE["grey"] for b in beta]
ax.bar(np.arange(1, p + 1), pip, color=cols, width=0.62)
for k in range(p):
    ax.text(k + 1, pip[k] + 0.025, f"{pip[k]:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(np.arange(1, p + 1))
ax.set_xticklabels([rf"$x_{k}$" for k in range(1, p + 1)])
ax.set_ylim(0, 1.1)
ax.set_ylabel("posterior inclusion probability")
ax.set_title("Inclusion probabilities", loc="left")
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.plot(grid, cont, color=PALETTE["blue"], label=r"BMA, continuous part")
ax.plot(grid, dens_full, color=PALETTE["red"], ls=(0, (4, 3)), lw=1.5, label="full model only")
ax.vlines(0, 0, 1 - pip[j], color=PALETTE["blue"], lw=3.2)
ax.plot([0], [1 - pip[j]], "o", color=PALETTE["blue"], ms=5)
ax.annotate(rf"mass {1 - pip[j]:.2f} at $0$", xy=(0, 1 - pip[j]), xytext=(-0.27, 2.6),
            fontsize=9.5, arrowprops=dict(arrowstyle="-", color=PALETTE["ink"], lw=0.6))
ax.axvline(beta[j], color=PALETTE["grey"], lw=1, ls=":")
ax.set_xlim(-0.3, 0.8)
ax.set_ylim(0, 3.6)
ax.set_xlabel(r"$\beta_6$")
ax.set_ylabel("posterior density")
ax.set_title(r"Model-averaged posterior of $\beta_6$", loc="left")
ax.legend(loc="upper right")
save(fig, __file__, "bma-posterior")

# ---- repeated experiment: out-of-sample mean log predictive density
reps, m_test = 300, 200
rng = np.random.default_rng(11)
diff_hpm, diff_full = [], []
for _ in range(reps):
    X, y = simulate(rng, n)
    Xt, yt = simulate(rng, m_test)
    out, w, xbar, ybar = fit(X, y, g=n)
    lp = log_pred(out, w, xbar, ybar, Xt, yt, n)
    top = lp.max(0)
    bma = np.mean(np.log(w @ np.exp(lp - top)) + top)
    diff_hpm.append(bma - lp[np.argmax(w)].mean())
    diff_full.append(bma - lp[-1].mean())
diff_hpm, diff_full = np.array(diff_hpm), np.array(diff_full)
print("BMA - HPM: mean %.4f, frac>0 %.3f" % (diff_hpm.mean(), (diff_hpm > 0).mean()))
print("BMA - full: mean %.4f, frac>0 %.3f" % (diff_full.mean(), (diff_full > 0).mean()))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
bins = np.arange(-0.10, 0.1451, 0.005)
for ax, d, title, col in [(axes[0], diff_hpm, "BMA minus highest-probability model", PALETTE["blue"]),
                          (axes[1], diff_full, "BMA minus full model", PALETTE["green"])]:
    ax.hist(d, bins=bins, color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.axvline(d.mean(), color=PALETTE["red"], lw=1.4, ls=(0, (4, 3)))
    ax.set_title(title, loc="left")
    ax.set_xlabel("difference in mean log predictive density")
    ax.text(0.97, 0.93, f"mean {d.mean():.3f}\npositive in {100 * (d > 0).mean():.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
axes[0].set_ylabel("replications")
save(fig, __file__, "bma-predictive")
