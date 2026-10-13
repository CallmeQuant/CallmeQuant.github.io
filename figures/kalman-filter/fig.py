"""Kalman filter for a constant-velocity model with noisy position measurements
(note: kalman-filter). (a) true position, observations and filtered mean with 95% bands;
(b) predicted and filtered variance of the position for the first 25 steps."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

use_style()
rng = np.random.default_rng(721)
T, dt, q, r = 60, 1.0, 0.02, 4.0
F = np.array([[1.0, dt], [0.0, 1.0]])
Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
H = np.array([[1.0, 0.0]])
R = np.array([[r]])
m0, P0 = np.array([0.0, 0.0]), np.diag([4.0, 0.25])

# simulate
z = np.zeros((T + 1, 2)); y = np.zeros(T)
z[0] = rng.multivariate_normal(m0, P0)
for t in range(1, T + 1):
    z[t] = F @ z[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
    y[t - 1] = (H @ z[t])[0] + rng.normal(0, np.sqrt(r))

# filter
m, P = m0.copy(), P0.copy()
mf = np.zeros((T, 2)); Pf = np.zeros((T, 2, 2)); Pp_pos = np.zeros(T)
for t in range(T):
    mp, Pp = F @ m, F @ P @ F.T + Q
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)
    m = mp + K @ (y[t] - H @ mp)
    P = Pp - K @ S @ K.T
    mf[t], Pf[t], Pp_pos[t] = m, P, Pp[0, 0]

t = np.arange(1, T + 1)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
sd = np.sqrt(Pf[:, 0, 0])
ax.fill_between(t, mf[:, 0] - 1.96 * sd, mf[:, 0] + 1.96 * sd, color=PALETTE["blue"], alpha=0.18, lw=0,
                label="95% band")
ax.scatter(t, y, s=9, color=PALETTE["grey"], label=r"observation $y_t$", zorder=2)
ax.plot(t, z[1:, 0], color=PALETTE["ink"], lw=1.2, ls="--", label=r"true position")
ax.plot(t, mf[:, 0], color=PALETTE["blue"], label=r"filtered mean $(\mu_{t\mid t})_1$")
ax.set_xlabel(r"time $t$")
ax.set_ylabel("position")
ax.set_title("(a) filtering the position", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

ax = axes[1]
k = 25
tt = np.repeat(np.arange(1, k + 1), 2)
vv = np.ravel(np.column_stack([Pp_pos[:k], Pf[:k, 0, 0]]))
ax.plot(tt, vv, color=PALETTE["grey"], lw=1.0)
ax.scatter(np.arange(1, k + 1), Pp_pos[:k], s=14, color=PALETTE["red"], label=r"predicted $(\Sigma_{t\mid t-1})_{11}$", zorder=3)
ax.scatter(np.arange(1, k + 1), Pf[:k, 0, 0], s=14, color=PALETTE["blue"], label=r"filtered $(\Sigma_{t\mid t})_{11}$", zorder=3)
ax.set_ylim(0, None)
ax.set_xlabel(r"time $t$")
ax.set_title("(b) position variance", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
save(fig, __file__, "tracking")
