"""Random Fourier features: Monte Carlo approximations of the Gaussian and Laplace kernels, and the error as
a function of the number of features (note: random Fourier features)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, CYCLE, save

use_style()
rng = np.random.default_rng(2007)
ELL = 1.0
delta = np.linspace(-6, 6, 601)
kernels = {
    "Gaussian": (lambda d: np.exp(-0.5 * d ** 2 / ELL ** 2), lambda size: rng.normal(0, 1 / ELL, size),
                 r"Gaussian kernel, $\omega \sim \mathcal{N}(0, \ell^{-2})$"),
    "Laplace": (lambda d: np.exp(-np.abs(d) / ELL), lambda size: rng.standard_cauchy(size) / ELL,
                r"Laplace kernel, $\omega \sim \mathrm{Cauchy}(0, \ell^{-1})$"),
}


def rff_kernel(d, omega, b):
    """(2/D) sum_j cos(omega_j x + b_j) cos(omega_j x' + b_j) with x = d, x' = 0."""
    return 2 * np.mean(np.cos(np.outer(d, omega) + b) * np.cos(b), axis=1)


# ---------------- Figure 1 -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
for ax, (name, (kf, sampler, title)) in zip(axes, kernels.items()):
    ax.plot(delta, kf(delta), color=PALETTE["ink"], lw=2.0, label="exact")
    for c, D in zip([PALETTE["gold"], PALETTE["blue"]], [20, 500]):
        om = sampler(D)
        b = rng.uniform(0, 2 * np.pi, D)
        ax.plot(delta, rff_kernel(delta, om, b), color=c, lw=1.2, label=rf"$D = {D}$")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.65, 1.3)
    ax.set_xlabel(r"$\delta = x - x'$")
axes[0].set_ylabel(r"$\hat k_D(\delta)$")
axes[0].legend(loc="upper left", fontsize=8.8)
save(fig, __file__, "kernel-approximations")

# ---------------- Figure 2 -----------------------------------------------------------------------
kf, sampler, _ = kernels["Gaussian"]
Ds = np.unique(np.geomspace(4, 4096, 12).astype(int) // 2 * 2)
reps = 400
d0 = 1.0
k0, k2 = kf(d0), kf(2 * d0)
var_phase = 1 + 0.5 * k2 - k0 ** 2          # per-feature variance, random-phase features
var_pair = 1 + k2 - 2 * k0 ** 2              # D features = D/2 (cos, sin) pairs, times D
grid = np.linspace(-4, 4, 161)
sd_phase, sd_pair, sup_phase, sup_pair = [], [], [], []
for D in Ds:
    e_ph, e_pr, s_ph, s_pr = [], [], [], []
    for _ in range(reps):
        om = sampler(D)
        b = rng.uniform(0, 2 * np.pi, D)
        e_ph.append(rff_kernel(np.array([d0]), om, b)[0] - k0)
        s_ph.append(np.max(np.abs(rff_kernel(grid, om, b) - kf(grid))))
        om2 = sampler(D // 2)
        e_pr.append(np.mean(np.cos(om2 * d0)) - k0)
        s_pr.append(np.max(np.abs(np.mean(np.cos(np.outer(grid, om2)), axis=1) - kf(grid))))
    sd_phase.append(np.std(e_ph)); sd_pair.append(np.std(e_pr))
    sup_phase.append(np.mean(s_ph)); sup_pair.append(np.mean(s_pr))

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)
ax = axes[0]
ax.loglog(Ds, sd_phase, "o", color=PALETTE["red"], ms=4.5, label="random phase (simulated)")
ax.loglog(Ds, np.sqrt(var_phase / Ds), color=PALETTE["red"], lw=1.2, label="random phase (formula)")
ax.loglog(Ds, sd_pair, "s", color=PALETTE["blue"], ms=4, label=r"$(\cos,\sin)$ pairs (simulated)")
ax.loglog(Ds, np.sqrt(var_pair / Ds), color=PALETTE["blue"], lw=1.2, label=r"$(\cos,\sin)$ pairs (formula)")
ax.set_xlabel(r"number of features $D$")
ax.set_ylabel("error")
ax.set_title(rf"Standard deviation of $\hat k_D(\delta)$, $\delta = {d0:g}$", loc="left", fontsize=10.5)
ax.legend(loc="lower left", fontsize=8.2)
ax = axes[1]
ax.loglog(Ds, sup_phase, "o-", color=PALETTE["red"], ms=4.5, lw=1.1, label="random phase")
ax.loglog(Ds, sup_pair, "s-", color=PALETTE["blue"], ms=4, lw=1.1, label=r"$(\cos,\sin)$ pairs")
ax.loglog(Ds, 2.5 / np.sqrt(Ds), color=PALETTE["grey"], ls="--", lw=1.0, label=r"slope $-1/2$")
ax.set_xlabel(r"number of features $D$")
ax.set_title(r"Mean of $\sup\, |\hat k_D - k|$ over $|\delta| \leq 4$", loc="left", fontsize=10.5)
ax.legend(loc="lower left", fontsize=8.2)
save(fig, __file__, "error-vs-features")
print("var_phase, var_pair at delta=1:", var_phase, var_pair)
