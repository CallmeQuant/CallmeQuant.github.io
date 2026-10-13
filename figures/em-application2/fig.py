"""EM versus stochastic EM (SEM) for a two-component univariate Gaussian mixture
(note: em-application2). (a) trajectories of the component means; (b) SEM iterates of the
second mean after burn-in, with the EM limit and the SEM average."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(113)
n = 400
z = rng.random(n) < 0.35
x = np.where(z, rng.normal(2.0, 0.8, n), rng.normal(-1.0, 1.2, n))


def resp(x, w, mu, s2):
    lp = np.log(w) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (x[:, None] - mu) ** 2 / s2
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def m_step(x, g):
    Nk = g.sum(0)
    mu = g.T @ x / Nk
    s2 = (g * (x[:, None] - mu) ** 2).sum(0) / Nk
    return Nk / len(x), mu, s2


T, burn = 3000, 300
start = (np.array([0.5, 0.5]), np.array([-0.01, 0.01]), np.array([3.0, 3.0]))

w, mu, s2 = start
em_mu = [np.sort(mu)]
for t in range(T):
    w, mu, s2 = m_step(x, resp(x, w, mu, s2))
    em_mu.append(np.sort(mu))
em_mu = np.array(em_mu)

w, mu, s2 = start
sem_mu = [np.sort(mu)]
for t in range(T):
    g = resp(x, w, mu, s2)
    while True:                          # S-step; redraw if a component gets fewer than 2 points
        lab = (rng.random(n)[:, None] > np.cumsum(g, axis=1)).sum(1)   # categorical draws
        onehot = np.eye(2)[lab]
        if onehot.sum(0).min() >= 2:
            break
    w, mu, s2 = m_step(x, onehot)
    sem_mu.append(np.sort(mu))   # order the components by their means
sem_mu = np.array(sem_mu)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
k = 500
for j, col in enumerate([PALETTE["blue"], PALETTE["red"]]):
    ax.plot(np.arange(k + 1), sem_mu[: k + 1, j], color=col, lw=0.9, alpha=0.9,
            label=fr"SEM $\mu_{{({j + 1})}}$")
    ax.plot(np.arange(k + 1), em_mu[: k + 1, j], color=PALETTE["ink"], lw=1.5,
            ls="--" if j else "-", label=fr"EM $\mu_{{({j + 1})}}$")
ax.set_xlabel(r"iteration $t$")
ax.set_title("(a) component means", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)

ax = axes[1]
draws = sem_mu[burn:, 1]   # larger of the two means
ax.hist(draws, bins=40, density=True, color=PALETTE["red"], alpha=0.35, lw=0)
ax.axvline(em_mu[-1, 1], color=PALETTE["ink"], lw=1.5, label="EM limit")
ax.axvline(draws.mean(), color=PALETTE["red"], lw=1.5, ls="--", label="SEM average")
ax.set_xlabel(r"$\mu_{(2)}$")
ax.set_ylabel("density")
ax.set_title(fr"(b) SEM iterates, $t > {burn}$", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, __file__, "em-vs-sem")
print("EM limit", em_mu[-1], "SEM mean", sem_mu[burn:].mean(0), "SEM sd", sem_mu[burn:].std(0))
print("EM mu at t=20,50,100", em_mu[[20, 50, 100]])
