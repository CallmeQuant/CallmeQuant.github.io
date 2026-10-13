"""EM for a two-component Poisson mixture on simulated counts (note: the EM algorithm).
Left: data and fitted mixture. Right: log-likelihood gap along EM iterations from three starts."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import logsumexp
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2023)
n, pi_true, lam_true = 500, np.array([0.6, 0.4]), np.array([2.0, 9.0])
z = rng.choice(2, size=n, p=pi_true)
x = rng.poisson(lam_true[z])


def loglik(x, pi, lam):
    return logsumexp(np.log(pi) + poisson.logpmf(x[:, None], lam), axis=1).sum()


def em(x, pi, lam, iters):
    pi, lam = np.array(pi, float), np.array(lam, float)
    ll = [loglik(x, pi, lam)]
    for _ in range(iters):
        logw = np.log(pi) + poisson.logpmf(x[:, None], lam)          # E-step
        r = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
        Nk = r.sum(axis=0)                                            # M-step
        pi, lam = Nk / x.size, (r * x[:, None]).sum(axis=0) / Nk
        ll.append(loglik(x, pi, lam))
    return pi, lam, np.array(ll)


starts = [((0.5, 0.5), (1.0, 3.0), "start A"), ((0.9, 0.1), (5.0, 6.0), "start B"),
          ((0.5, 0.5), (4.0, 4.0), r"start C ($\lambda_1 = \lambda_2$)")]
pi_hat, lam_hat, _ = em(x, starts[0][0], starts[0][1], 2000)
ll_hat = loglik(x, pi_hat, lam_hat)
print("estimates", pi_hat, lam_hat, "loglik", ll_hat)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
ax = axes[0]
k = np.arange(0, x.max() + 2)
counts = np.bincount(x, minlength=k.size) / n
ax.bar(k, counts, width=0.85, color=PALETTE["light"], edgecolor=PALETTE["grey"], lw=0.5, label="data")
for j in range(2):
    ax.plot(k, pi_hat[j] * poisson.pmf(k, lam_hat[j]), color=CYCLE[j], lw=1.2, ls=(0, (3, 2)),
            label=rf"$\hat\pi_{j+1}\,\mathrm{{Pois}}(\hat\lambda_{j+1})$")
ax.plot(k, (pi_hat * poisson.pmf(k[:, None], lam_hat)).sum(1), color=PALETTE["ink"], lw=1.5, marker="o",
        ms=2.8, label="fitted mixture")
ax.set_xlim(-0.8, x.max() + 1.5)
ax.set_xlabel(r"count $x$"); ax.set_ylabel("relative frequency")
ax.set_title(r"(a) $n = 500$ counts and the EM fit", loc="left")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
for j, (p0, l0, lab) in enumerate(starts):
    _, lam_end, ll = em(x, p0, l0, 60)
    gap = ll_hat - ll
    ax.plot(np.arange(ll.size), np.maximum(gap, 1e-12), color=CYCLE[j], lw=1.6, label=lab)
    print(lab, "final rates", lam_end, "final gap", gap[-1])
ax.set_yscale("log")
ax.set_ylim(1e-9, 3e3)
ax.set_xlim(0, 30)
ax.set_xlabel(r"iteration $t$")
ax.set_ylabel(r"$\ell(\hat\theta) - \ell(\theta_t)$")
ax.set_title("(b) log-likelihood gap", loc="left")
ax.legend(loc="center right", fontsize=9)
save(fig, __file__, "poisson-mixture-em")
