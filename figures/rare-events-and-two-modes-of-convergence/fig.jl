# Independent Bernoulli(p_n) indicators: times at which X_n = 1.
# Packages: CairoMakie, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random

set_blog_theme!()
rng = MersenneTwister(20191111)
N, paths = 10_000, 10
n = collect(1:N)
cases = [
    (L"p_n = 1/n:\ \sum p_n = \infty,\ \text{ones occur infinitely often}", 1 ./ n, PALETTE.blue),
    (L"p_n = 1/n^2:\ \sum p_n < \infty,\ \text{ones eventually stop}", 1 ./ n .^ 2, PALETTE.red),
]

fig = Figure(size = (680, 440))
axes = Axis[]
for (i, (title, p, col)) in enumerate(cases)
    ax = Axis(fig[i, 1]; title = title, ylabel = "path", xscale = log10,
              yticks = [1, 5, 10], ygridvisible = false,
              xlabel = i == 2 ? "time n (log scale)" : "")
    for j in 1:paths
        hits = n[rand(rng, N) .< p]
        scatter!(ax, hits, fill(j, length(hits)); marker = :vline, markersize = 14, color = col)
    end
    xlims!(ax, 0.8, N * 1.25); ylims!(ax, 0.3, paths + 0.7)
    push!(axes, ax)
end
linkxaxes!(axes...)
hidexdecorations!(axes[1]; grid = false)
save_fig(fig, @__DIR__, "indicator-paths")
