"""Slice sampling with stepping out and shrinkage on a two-component normal mixture
(note: slice sampling)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, save

use_style()

# Unnormalized target: 0.55 N(-2.5, 0.5^2) + 0.45 N(2.5, 0.7^2) (normalized here, but the sampler never uses it).
f = lambda x: 0.55 * norm.pdf(x, -2.5, 0.5) + 0.45 * norm.pdf(x, 2.5, 0.7)


def slice_step(x, w, rng, record=False):
    """One slice-sampling update with unlimited stepping out and shrinkage."""
    u = rng.uniform(0, f(x))
    L = x - w * rng.random(); R = L + w
    steps = [(L, R)]
    while f(L) > u:
        L -= w; steps.append((L, R))
    while f(R) > u:
        R += w; steps.append((L, R))
    rejected = []
    while True:
        y = rng.uniform(L, R)
        if f(y) > u:
            break
        rejected.append(y)
        if y < x: L = y
        else: R = y
    return (y, u, steps, rejected) if record else y


def run(x0, w, T, rng):
    out = np.empty(T); x = x0
    for t in range(T):
        x = slice_step(x, w, rng); out[t] = x
    return out


# ---- Figure 1: anatomy of one update
xs = np.linspace(-5, 5, 1000)
x0 = -2.05
rng = np.random.default_rng(3)
while True:  # draw a step whose level cuts both modes, so the slice has two components
    st = np.random.default_rng(rng.integers(1 << 31))
    y, u, steps, rej = slice_step(x0, 1.0, st, record=True)
    if 0.08 < u < 0.16 and len(rej) >= 1:
        break

fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.fill_between(xs, f(xs), color=PALETTE["light"], lw=0)
ax.plot(xs, f(xs), color=PALETTE["ink"], lw=1.4)
ax.plot([x0, x0], [0, f(x0)], color=PALETTE["grey"], lw=1, ls="--")
ax.plot(x0, u, "o", color=PALETTE["ink"], ms=5)
ax.axhline(u, color=PALETTE["grey"], lw=0.8)
inside = f(xs) > u
seg = np.where(inside, u, np.nan)
ax.plot(xs, seg, color=PALETTE["blue"], lw=4, solid_capstyle="butt", label=r"slice $S_u = \{x : f(x) > u\}$")
L, R = steps[-1]
for k, (l, r) in enumerate(steps):
    yy = -0.035 - 0.022 * k
    ax.plot([l, r], [yy, yy], color=PALETTE["gold"], lw=1.4)
    ax.plot([l, r], [yy, yy], "|", color=PALETTE["gold"], ms=7)
ax.plot([], [], color=PALETTE["gold"], lw=1.4, label=r"stepping out, $w = 1$")
ax.plot(rej, [u] * len(rej), "x", color=PALETTE["red"], ms=8, mew=2, label="rejected, interval shrinks")
ax.plot(y, u, "o", color=PALETTE["green"], ms=7, label=r"accepted $x'$")
ax.annotate(r"$x$", (x0, 0), xytext=(x0 + 0.08, 0.012), color=PALETTE["ink"])
ax.annotate(r"$u \sim \mathrm{U}(0, f(x))$", (x0, u), xytext=(-1.1, 0.31),
            arrowprops=dict(arrowstyle="-", color=PALETTE["grey"], lw=0.8))
ax.set_xlim(-5, 5); ax.set_ylim(-0.035 - 0.022 * len(steps), 0.5)
ax.set_xlabel("$x$"); ax.set_ylabel(r"$f(x)$")
ax.set_yticks([0, 0.2, 0.4])
ax.legend(loc="upper right")
save(fig, __file__, "one-update")

# ---- Figure 2: output for two interval widths
T = 20_000
rng = np.random.default_rng(42)
runs = {w: run(-2.5, w, T, rng) for w in (0.3, 4.0)}
fig = plt.figure(figsize=(7.8, 4.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
for i, (w, col) in enumerate([(0.3, PALETTE["red"]), (4.0, PALETTE["blue"])]):
    x = runs[w]
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(np.arange(T), x, color=col, lw=0.35)
    ax.set_ylim(-5, 5); ax.set_xlim(0, T); ax.set_xticks([0, 5000, 10000, 15000, 20000]); ax.set_xticklabels(["0", "5000", "10000", "15000", "20000"]); ax.set_ylabel("$x_t$")
    frac = (x > 0).mean()
    ax.set_title(rf"$w = {w}$: fraction of draws with $x_t > 0$ is {frac:.2f} (target 0.45)",
                 loc="left", fontsize=10.5)
    if i == 0:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("iteration $t$")
    axh = fig.add_subplot(gs[i, 1])
    axh.hist(x, bins=np.linspace(-5, 5, 61), density=True, color=col, alpha=0.45, lw=0)
    axh.plot(xs, f(xs), color=PALETTE["ink"], lw=1.2)
    axh.set_xlim(-5, 5); axh.set_ylim(0, 0.62)
    if i == 0:
        axh.tick_params(labelbottom=False)
        axh.set_title("histogram and target", loc="left", fontsize=10.5)
    else:
        axh.set_xlabel("$x$")
save(fig, __file__, "interval-width")
