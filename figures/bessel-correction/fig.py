"""Mean squared error of SS/d for the divisors d = n-1, n, n+1 and the MSE-optimal divisor,
relative to the unbiased choice d = n-1, for three values of the kurtosis (note: Bessel's correction).
Exact formulas; a Monte Carlo check is printed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save


def mse(d, n, kappa):
    """MSE of SS/d for sigma^2 = 1, where SS = sum (X_i - Xbar)^2 and kappa = mu_4 / sigma^4."""
    ess2 = (n - 1) ** 2 * (kappa / n - (n - 3) / (n * (n - 1)) + 1)
    return ess2 / d**2 - 2 * (n - 1) / d + 1


use_style()
n = np.arange(2, 31)
cases = [(1, r"$\kappa = 1$ (symmetric $\pm 1$)"), (3, r"$\kappa = 3$ (normal)"), (9, r"$\kappa = 9$ (exponential)")]

fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
for ax, (kappa, title) in zip(axes, cases):
    base = mse(n - 1, n, kappa)
    dstar = n + 1 + (n - 1) * (kappa - 3) / n
    ax.axhline(1, color=PALETTE["grey"], lw=1.0, ls=(0, (4, 3)))
    ax.plot(n, mse(n, n, kappa) / base, color=PALETTE["blue"], label=r"$d = n$")
    ax.plot(n, mse(n + 1, n, kappa) / base, color=PALETTE["red"], label=r"$d = n+1$")
    ax.plot(n, mse(dstar, n, kappa) / base, color=PALETTE["ink"], lw=1.2, ls=(0, (1, 1.5)), label=r"$d = d^\ast$")
    ax.set_title(title, loc="left")
    ax.set_xlabel(r"sample size $n$")
    ax.set_xlim(2, 30)
    ax.set_ylim(0, 3.2)
axes[0].set_ylabel(r"MSE$(d)\,/\,$MSE$(n-1)$")
axes[2].legend(loc="upper right")
save(fig, __file__, "mse-by-divisor")

# Monte Carlo check of the formula at n = 5
rng = np.random.default_rng(23)
gens = {1: lambda s: rng.integers(0, 2, s) * 2.0 - 1.0,
        3: lambda s: rng.standard_normal(s),
        9: lambda s: rng.exponential(size=s) - 1.0}
for kappa, g in gens.items():
    X = g((200_000, 5))
    SS = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    for d in (4, 5, 6):
        print(kappa, d, round(float(np.mean((SS / d - 1) ** 2)), 4), round(float(mse(d, 5, kappa)), 4))
