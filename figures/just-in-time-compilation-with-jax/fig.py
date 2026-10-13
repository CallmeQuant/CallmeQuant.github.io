"""Timing NumPy, eager JAX and jit-compiled JAX.

Timings are machine-specific. If data.csv is absent this script measures them (CPU, float64) and
writes data.csv; the R and Julia scripts plot the same file. Delete data.csv to re-measure.
Packages: numpy, matplotlib, jax (pip install jax).
"""
import sys, pathlib, time, math, platform, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_style"))
import numpy as np
import matplotlib.pyplot as plt
from style import use_style, PALETTE, save

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data.csv"


def measure():
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def per_call(fn, arg, min_total=0.2, batches=7):
        """Smallest time per call over batches (robust to interference from other processes); jax results are blocked on before stopping the clock."""
        block = lambda r: r.block_until_ready() if hasattr(r, "block_until_ready") else r
        block(fn(arg))                                    # warm-up (includes compilation for jit)
        reps, t = 1, 0.0
        while True:
            t0 = time.perf_counter()
            for _ in range(reps):
                block(fn(arg))
            t = time.perf_counter() - t0
            if t > min_total / batches or reps > 1e6:
                break
            reps *= 4
        out = []
        for _ in range(batches):
            t0 = time.perf_counter()
            for _ in range(reps):
                block(fn(arg))
            out.append((time.perf_counter() - t0) / reps)
        return float(np.min(out))

    rows = []
    # Experiment 1: a chain of elementwise operations followed by a reduction.
    f_np = lambda x: np.sum(np.exp(-0.5 * x**2) * np.cos(3 * x) + x * np.tanh(x))
    f_jx = lambda x: jnp.sum(jnp.exp(-0.5 * x**2) * jnp.cos(3 * x) + x * jnp.tanh(x))
    sizes = np.unique(np.round(np.logspace(1, 7, 13)).astype(int))
    rng = np.random.default_rng(0)
    for n in sizes:
        x = rng.standard_normal(n); xj = jnp.asarray(x)
        rows.append(("elementwise", "numpy", n, per_call(f_np, x)))
        rows.append(("elementwise", "jax_eager", n, per_call(f_jx, xj)))
        firsts = []
        for _ in range(3):                                # smallest of three cold first calls
            jax.clear_caches()
            f_jit = jax.jit(f_jx)                         # empty cache: the first call traces and compiles
            t0 = time.perf_counter(); f_jit(xj).block_until_ready(); firsts.append(time.perf_counter() - t0)
        first = min(firsts)
        rows.append(("elementwise", "jax_jit_first_call", n, first))
        rows.append(("elementwise", "jax_jit", n, per_call(f_jit, xj)))
        print("elementwise", n, rows[-4:])

    # Experiment 2: a nonlinear recursion x_t = tanh(0.9 x_{t-1}) + e_t that cannot be vectorized.
    def rec_python(e):
        x, out = 0.0, [0.0] * len(e)
        for t, et in enumerate(e):
            x = math.tanh(0.9 * x) + et
            out[t] = x
        return out

    def rec_numpy(e):
        out = np.empty_like(e); x = 0.0
        for t in range(e.size):
            x = np.tanh(0.9 * x) + e[t]
            out[t] = x
        return out

    def step(x, et):
        x = jnp.tanh(0.9 * x) + et
        return x, x

    rec_scan = jax.jit(lambda e: jax.lax.scan(step, 0.0, e)[1])

    def rec_jax_eager(e):
        x, out = jnp.asarray(0.0), []
        for t in range(e.shape[0]):
            x = jnp.tanh(0.9 * x) + e[t]
            out.append(x)
        return jnp.stack(out)

    for T in [100, 1000, 10_000, 100_000, 1_000_000]:
        e = 0.1 * rng.standard_normal(T); el = e.tolist(); ej = jnp.asarray(e)
        rows.append(("recursion", "python_loop", T, per_call(rec_python, el, min_total=0.1, batches=3)))
        rows.append(("recursion", "numpy_loop", T, per_call(rec_numpy, e, min_total=0.1, batches=3)))
        if T <= 10_000:
            rows.append(("recursion", "jax_eager_loop", T, per_call(rec_jax_eager, ej, min_total=0.1, batches=3)))
        rows.append(("recursion", "jax_scan_jit", T, per_call(rec_scan, ej)))
        print("recursion", T, rows[-3:])

    with open(DATA, "w", newline="") as fh:
        fh.write(f"# measured with jax {jax.__version__}, numpy {np.__version__}, python {platform.python_version()}, "
                 f"{platform.processor() or platform.machine()}, float64, CPU\n")
        w = csv.writer(fh)
        w.writerow(["experiment", "method", "size", "seconds"])
        w.writerows(rows)


if not DATA.exists():
    measure()

# ---- plotting from data.csv
d = {}
with open(DATA) as fh:
    for r in csv.DictReader(line for line in fh if not line.startswith("#")):
        d.setdefault((r["experiment"], r["method"]), []).append((int(r["size"]), float(r["seconds"])))
get = lambda e, m: np.array(sorted(d[(e, m)])).T

use_style()
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
ax = axes[0]
for m, col, lab, ls in [("numpy", PALETTE["red"], "NumPy", "-"), ("jax_eager", PALETTE["gold"], "JAX, op by op", "-"),
                        ("jax_jit", PALETTE["blue"], "JAX, jit (compiled)", "-"),
                        ("jax_jit_first_call", PALETTE["grey"], "jit, first call (trace + compile)", "--")]:
    n, s = get("elementwise", m)
    ax.plot(n, s, "o" + ls, color=col, ms=3.5, lw=1.3, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("array length $n$"); ax.set_ylabel("seconds per call")
ax.set_title("Elementwise expression and sum", loc="left")
ax.set_ylim(4e-6, 200)
ax.legend(loc="upper left", fontsize=8.6)

ax = axes[1]
for m, col, lab in [("python_loop", PALETTE["red"], "Python loop, floats"),
                    ("numpy_loop", PALETTE["violet"], "Python loop, NumPy scalars"),
                    ("jax_eager_loop", PALETTE["gold"], "Python loop, JAX op by op"),
                    ("jax_scan_jit", PALETTE["blue"], r"lax.scan under jit")]:
    T, s = get("recursion", m)
    ax.plot(T, s, "o-", color=col, ms=3.5, lw=1.3, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of time steps $T$"); ax.set_ylabel("seconds per call")
ax.set_title(r"Recursion $x_t = \tanh(0.9\,x_{t-1}) + e_t$", loc="left")
ax.set_ylim(4e-6, 200)
ax.legend(loc="upper right", fontsize=8.6)
save(fig, __file__, "timings")
