"""Union bound for the maximum of m = 20 equicorrelated standard normals (note: Boole's inequality).
Left: P(max Z_i > t) for correlations 0 and 1, the union bound and the second Bonferroni lower bound
(independent case). Right: ratio of the union bound to the exact probability for several correlations."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, CYCLE, save

use_style()
m, alpha = 20, 0.05
t = np.linspace(0.0, 4.5, 451)
tail = norm.sf(t)

# Gauss-Hermite rule for E over W ~ N(0,1)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(120)
gh_w = gh_w / gh_w.sum()


def p_max_exceeds(t, rho):
    """P(max_i Z_i > t) for Z_i = sqrt(rho) W + sqrt(1 - rho) E_i, i = 1..m."""
    if rho == 1.0:
        return norm.sf(t)
    if rho == 0.0:
        return -np.expm1(m * norm.logcdf(t))
    a = (t[:, None] - np.sqrt(rho) * gh_x[None, :]) / np.sqrt(1 - rho)
    return 1.0 - np.exp(m * norm.logcdf(a)) @ gh_w


union = m * tail
bonf2 = m * tail - m * (m - 1) / 2 * tail**2
t_bonf = norm.isf(alpha / m)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
ax = axes[0]
ax.plot(t, np.minimum(union, 1), color=PALETTE["ink"], lw=1.4, ls=(0, (4, 2)), label=r"union bound $m\,\bar\Phi(t)$")
ax.plot(t, p_max_exceeds(t, 0.0), color=CYCLE[0], label=r"exact, $\rho = 0$")
ax.plot(t, p_max_exceeds(t, 1.0), color=CYCLE[3], label=r"exact, $\rho = 1$")
pos = bonf2 > 0
ax.plot(t[pos], bonf2[pos], color=PALETTE["grey"], lw=1.4, ls=(0, (1, 1.5)), label=r"Bonferroni lower bound, $\rho = 0$")
ax.axvline(t_bonf, color=PALETTE["violet"], lw=0.9, ls=(0, (2, 2)))
ax.annotate(r"$t = \bar\Phi^{-1}(\alpha/m)$", xy=(t_bonf, 0.4), xytext=(t_bonf + 0.1, 0.4),
            color=PALETTE["violet"], fontsize=9.5, va="center")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.6)
ax.set_xlim(0, 4.5)
ax.set_xlabel(r"threshold $t$")
ax.set_title(r"$\mathrm{P}(\mathrm{max}_i\, Z_i > t)$, $m = 20$", loc="left")
ax.legend(loc="lower left", fontsize=8.8)

ax = axes[1]
for k, rho in enumerate([0.0, 0.5, 0.9, 1.0]):
    ax.plot(t, union / p_max_exceeds(t, rho), color=CYCLE[k], label=rf"$\rho = {rho:g}$")
ax.axhline(1, color=PALETTE["grey"], lw=1.0, ls=(0, (4, 3)))
ax.vlines(t_bonf, 0.8, 28, color=PALETTE["violet"], lw=0.9, ls=(0, (2, 2)))
ax.set_yscale("log")
ax.set_xlim(0, 4.5)
ax.set_ylim(0.8, 70)
ax.set_yticks([1, 2, 5, 10, 20])
ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.set_xlabel(r"threshold $t$")
ax.set_title("union bound / exact probability", loc="left")
ax.legend(loc="upper center", ncols=4, columnspacing=1.2, handlelength=1.4)
save(fig, __file__, "union-bound-maximum")

for rho in (0.0, 0.5, 0.9, 1.0):
    print(rho, "P(max > t_bonf) =", float(p_max_exceeds(np.array([t_bonf]), rho)[0]))
# Monte Carlo check for rho = 0.5
rng = np.random.default_rng(11)
W = rng.standard_normal(200_000)
Z = np.sqrt(0.5) * W[:, None] + np.sqrt(0.5) * rng.standard_normal((200_000, m))
print("MC rho=0.5 at t=2:", (Z.max(1) > 2).mean(), float(p_max_exceeds(np.array([2.0]), 0.5)[0]))
