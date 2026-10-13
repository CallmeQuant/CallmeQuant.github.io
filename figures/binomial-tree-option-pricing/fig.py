"""Binomial (Cox-Ross-Rubinstein) option pricing (note: pricing by replication on a binomial
tree). Produces two figures:
  tree.svg        - four-period tree for an American put, with early-exercise nodes marked
  convergence.svg - CRR call price minus Black-Scholes against N; American versus European put
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from style import use_style, PALETTE, save

use_style()

# ---------------------------------------------------------------- tree (one-period units)
S0, K, u, R, N = 100.0, 100.0, 1.15, 1.01, 4
d = 1 / u
q = (R - d) / (u - d)
S = {(n, k): S0 * u**k * d**(n - k) for n in range(N + 1) for k in range(n + 1)}
V, ex = {}, {}
for k in range(N + 1):
    V[N, k] = max(K - S[N, k], 0.0); ex[N, k] = V[N, k] > 0
for n in range(N - 1, -1, -1):
    for k in range(n + 1):
        cont = (q * V[n + 1, k + 1] + (1 - q) * V[n + 1, k]) / R
        stop = max(K - S[n, k], 0.0)
        V[n, k] = max(cont, stop); ex[n, k] = stop > cont

fig, ax = plt.subplots(figsize=(7.2, 4.0))
pos = {(n, k): (n, 2 * k - n) for (n, k) in S}
for (n, k), (x, y) in pos.items():
    if n < N:
        for kk in (k, k + 1):
            x2, y2 = pos[n + 1, kk]
            ax.plot([x, x2], [y, y2], color=PALETTE["light"], lw=1.4, zorder=1)
for (n, k), (x, y) in pos.items():
    col = PALETTE["red"] if ex[n, k] and n < N else PALETTE["blue"]
    face = col if ex[n, k] and n < N else "white"
    ax.scatter([x], [y], s=1350, marker="o", facecolor=face, edgecolor=col, lw=1.3, zorder=2)
    tc = "white" if ex[n, k] and n < N else PALETTE["ink"]
    ax.text(x, y + 0.2, f"{S[n, k]:.1f}", ha="center", va="center", fontsize=9, color=tc, zorder=3)
    ax.text(x, y - 0.22, f"{V[n, k]:.2f}", ha="center", va="center", fontsize=9,
            color=tc, fontweight="bold", zorder=3)
ax.set_xlim(-0.5, N + 0.5); ax.set_ylim(-N - 0.8, N + 0.8)
ax.set_xticks(range(N + 1)); ax.set_yticks([])
ax.set_xlabel(r"period $n$")
ax.grid(False); ax.spines["left"].set_visible(False)
ax.set_title(rf"American put, $K = {K:.0f}$, $u = {u}$, $d = 1/u$, $R = {R}$:"
             r"  node shows $S_n$ (top) and $V_n$ (bottom)", loc="left", fontsize=10.5)
ax.scatter([], [], s=60, facecolor=PALETTE["red"], edgecolor=PALETTE["red"],
           label="early exercise optimal")
ax.scatter([], [], s=60, facecolor="white", edgecolor=PALETTE["blue"], label="continue (or expiry)")
ax.legend(loc="upper left", scatterpoints=1)
save(fig, __file__, "tree")


# ------------------------------------------------------------ convergence to Black-Scholes
def crr(S0, K, r, sigma, T, N, kind="call", american=False):
    """CRR price; S0 may be an array. Returns array of prices."""
    S0 = np.atleast_1d(np.asarray(S0, float))
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u; R = np.exp(r * dt)
    q = (R - d) / (u - d)
    k = np.arange(N + 1)
    ST = S0[:, None] * u**k * d**(N - k)
    pay = (lambda s: np.maximum(s - K, 0)) if kind == "call" else (lambda s: np.maximum(K - s, 0))
    V = pay(ST)
    for n in range(N - 1, -1, -1):
        V = (q * V[:, 1:n + 2] + (1 - q) * V[:, :n + 1]) / R
        if american:
            k = np.arange(n + 1)
            V = np.maximum(V, pay(S0[:, None] * u**k * d**(n - k)))
    return V[:, 0]


def bs(S0, K, r, sigma, T, kind="call"):
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


r, sigma, T = 0.05, 0.2, 1.0
Ns = np.arange(5, 301)
err = np.array([crr(100.0, 105.0, r, sigma, T, n)[0] for n in Ns]) - bs(100.0, 105.0, r, sigma, T)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.8, 3.4))
a0.axhline(0, color=PALETTE["ink"], lw=0.7)
for par, col, lab in ((0, PALETTE["blue"], r"$N$ even"), (1, PALETTE["gold"], r"$N$ odd")):
    sel = Ns % 2 == par
    a0.plot(Ns[sel], Ns[sel] * err[sel], "o-", ms=2.2, lw=0.6, color=col, label=lab)
a0.set_xlim(0, 300); a0.set_ylim(-2.6, 3.1)
a0.set_xlabel(r"number of periods $N$")
a0.set_title(r"call: $N\,(C_N - C_{\mathrm{BS}})$, $S_0 = 100$, $K = 105$", loc="left")
a0.legend(loc="upper right", ncol=2)

sig2, Kp, Nam = 0.3, 100.0, 500
s = np.linspace(50, 150, 201)
am = crr(s, Kp, r, sig2, T, Nam, kind="put", american=True)
eu = bs(s, Kp, r, sig2, T, kind="put")
intrinsic = np.maximum(Kp - s, 0)
s_star = s[am - intrinsic < 1e-6].max()
a1.plot(s, intrinsic, color=PALETTE["grey"], lw=1.2, label=r"payoff $(K - S_0)^+$")
a1.plot(s, eu, color=PALETTE["blue"], label="European (Black–Scholes)")
a1.plot(s, am, color=PALETTE["red"], label=rf"American (CRR, $N = {Nam}$)")
a1.axvline(s_star, color=PALETTE["ink"], ls=":", lw=1.0)
a1.text(s_star - 1.5, 8, rf"$S^* \approx {s_star:.1f}$", fontsize=9.5, ha="right")
a1.set_xlim(50, 150); a1.set_ylim(0, 52)
a1.set_xlabel(r"spot price $S_0$")
a1.set_title(rf"put value, $K = {Kp:.0f}$, $\sigma = {sig2}$", loc="left")
a1.legend(loc="upper right")
save(fig, __file__, "convergence")
print("q tree", q, "S* =", s_star, "max |err| N>=100:", np.abs(err[Ns >= 100]).max(),
      "max N|err|:", (Ns * np.abs(err)).max())
