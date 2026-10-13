"""Dirichlet process mixture of bivariate normals (known covariance sigma^2 I, base measure
N(mu0, tau0^2 I)) fitted by collapsed Gibbs sampling (Neal's Algorithm 3): data and one posterior
partition, trace of the number of clusters, and prior versus posterior of K (note: DP mixtures)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2000)

# ---- data: four clusters, sd 0.5
centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.5, 3.5]])
props = np.array([0.35, 0.30, 0.20, 0.15])
n, sigma = 300, 0.5
labels = rng.choice(4, size=n, p=props)
X = centers[labels] + sigma * rng.standard_normal((n, 2))

alpha, tau0 = 1.0, 3.0
mu0 = X.mean(0)


def log_pred(x, cnt, sm):
    """log N(x; m_S, (sigma^2 + s_S^2) I) for clusters with counts cnt and coordinate sums sm."""
    s2 = 1.0 / (1.0 / tau0**2 + cnt / sigma**2)
    m = s2[:, None] * (mu0 / tau0**2 + sm / sigma**2)
    v = sigma**2 + s2
    return -np.log(2 * np.pi * v) - ((x - m) ** 2).sum(1) / (2 * v)


def gibbs(X, sweeps, rng):
    z = np.zeros(len(X), dtype=int)          # start with everything in one cluster
    cnt = [len(X)]; sm = [X.sum(0)]
    trace, keep = [], []
    for it in range(sweeps):
        for i in rng.permutation(len(X)):
            k = z[i]; cnt[k] -= 1; sm[k] = sm[k] - X[i]
            if cnt[k] == 0:                    # remove the emptied cluster
                cnt.pop(k); sm.pop(k); z[z > k] -= 1
            c, s = np.array(cnt, float), np.array(sm)
            lp = np.append(np.log(c) + log_pred(X[i], c, s),
                           np.log(alpha) + log_pred(X[i], np.zeros(1), np.zeros((1, 2))))
            p = np.exp(lp - lp.max()); p /= p.sum()
            k = rng.choice(len(p), p=p)
            if k == len(cnt):
                cnt.append(0); sm.append(np.zeros(2))
            z[i] = k; cnt[k] += 1; sm[k] = sm[k] + X[i]
        trace.append(len(cnt)); keep.append(z.copy())
    return np.array(trace), keep


sweeps, burn = 400, 100
trace, keep = gibbs(X, sweeps, rng)
z_last = keep[-1]
print("posterior of K:", {k: round(np.mean(trace[burn:] == k), 3) for k in np.unique(trace[burn:])})

# ---- Figure 1: data with true labels and one posterior partition
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharex=True, sharey=True)
for ax, lab, title in [(axes[0], labels, "Data, coloured by true component"),
                       (axes[1], z_last, f"Partition at sweep {sweeps}")]:
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    for r, k in enumerate(order):
        pts = X[lab == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=11, color=CYCLE[r % len(CYCLE)], alpha=0.85, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel(r"$x_1$")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$x_2$")
save(fig, __file__, "dpmm-partition")

# ---- Figure 2: trace of K, and prior versus posterior of K
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.plot(np.arange(1, sweeps + 1), trace, color=PALETTE["blue"], lw=1.0, drawstyle="steps-mid")
ax.axvline(burn, color=PALETTE["grey"], lw=0.9, ls=(0, (4, 3)))
ax.text(burn + 6, 10.2, "burn-in ends", fontsize=9, color=PALETTE["grey"])
ax.set_xlim(0, sweeps)
ax.set_ylim(0, 11)
ax.set_xlabel("sweep")
ax.set_ylabel(r"number of clusters $K$")
ax.set_title("Trace of the number of clusters", loc="left")

# prior of K_n: sum of independent Bernoulli(alpha / (alpha + i - 1))
pk = np.array([1.0])
for i in range(1, n + 1):
    q = alpha / (alpha + i - 1)
    pk = np.append(pk * (1 - q), 0) + np.append(0, pk * q)
ks = np.arange(1, 16)
post = np.array([np.mean(trace[burn:] == k) for k in ks])
ax = axes[1]
ax.bar(ks - 0.2, pk[ks], width=0.4, color=PALETTE["grey"], label="prior")
ax.bar(ks + 0.2, post, width=0.4, color=PALETTE["blue"], label="posterior")
ax.set_xlim(0.3, 15.7)
ax.set_xticks([1, 4, 7, 10, 13])
ax.set_xlabel(r"$K$")
ax.set_ylabel("probability")
ax.set_title(r"Prior and posterior of $K$, $\alpha = 1$", loc="left")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, __file__, "dpmm-clusters")
