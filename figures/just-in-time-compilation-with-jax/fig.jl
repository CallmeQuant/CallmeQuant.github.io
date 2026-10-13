# Timing NumPy, eager JAX and jit-compiled JAX: plots the timings measured by fig.py (data.csv).
# Packages: CairoMakie, DelimitedFiles
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using DelimitedFiles

set_blog_theme!()
raw = readdlm(joinpath(@__DIR__, "data.csv"), ','; comments = true, comment_char = '#')
rows = raw[2:end, :]                                  # drop the header row
function series(exp, method)
    sel = (rows[:, 1] .== exp) .& (rows[:, 2] .== method)
    n = Float64.(rows[sel, 3]); s = Float64.(rows[sel, 4])
    p = sortperm(n)
    return n[p], s[p]
end

fig = Figure(size = (780, 350))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = L"\text{array length } n",
          ylabel = "seconds per call", title = "Elementwise expression and sum")
for (m, col, lab, ls) in [("numpy", PALETTE.red, "NumPy", :solid), ("jax_eager", PALETTE.gold, "JAX, op by op", :solid),
                          ("jax_jit", PALETTE.blue, "JAX, jit (compiled)", :solid),
                          ("jax_jit_first_call", PALETTE.grey, "jit, first call (trace + compile)", :dash)]
    n, s = series("elementwise", m)
    scatterlines!(ax, n, s; color = col, linestyle = ls, markersize = 6, label = lab)
end
ylims!(ax, 4e-6, 200)
axislegend(ax; position = :lt, labelsize = 11)

ax = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{number of time steps } T",
          ylabel = "seconds per call", title = L"\text{Recursion } x_t = \tanh(0.9\,x_{t-1}) + e_t")
for (m, col, lab) in [("python_loop", PALETTE.red, "Python loop, floats"),
                      ("numpy_loop", PALETTE.violet, "Python loop, NumPy scalars"),
                      ("jax_eager_loop", PALETTE.gold, "Python loop, JAX op by op"),
                      ("jax_scan_jit", PALETTE.blue, "lax.scan under jit")]
    T, s = series("recursion", m)
    scatterlines!(ax, T, s; color = col, markersize = 6, label = lab)
end
ylims!(ax, 4e-6, 200)
axislegend(ax; position = :rt, labelsize = 11)
save_fig(fig, @__DIR__, "timings")
