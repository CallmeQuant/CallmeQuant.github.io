# Dirichlet process DP(alpha, N(0,1)): random distribution functions for three concentration
# parameters (stick-breaking), stick-breaking weights, and the number of distinct values in a
# Polya urn sequence.
# Packages: CairoMakie, Distributions, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random

set_blog_theme!()

function stick_breaking(rng, alpha; tol = 1e-10)
    w = Float64[]; atoms = Float64[]; rest = 1.0
    while rest > tol
        v = rand(rng, Beta(1.0, alpha))
        push!(w, rest * v); push!(atoms, randn(rng)); rest *= 1 - v
    end
    return w, atoms
end

# ---- Figure 1
rng = MersenneTwister(1973)
x = range(-3, 3; length = 1201)
fig = Figure(size = (800, 290))
axs = Axis[]
for (j, alpha) in enumerate([1, 10, 100])
    ax = Axis(fig[1, j]; xlabel = L"x", ylabel = j == 1 ? L"P((-\infty, x])" : "", title = L"\alpha = %$alpha")
    for _ in 1:8
        w, a = stick_breaking(rng, alpha)
        o = sortperm(a)
        stairs!(ax, vcat(-3.0, a[o], 3.0), vcat(0.0, cumsum(w[o]), sum(w)); step = :post,
                color = (PALETTE.blue, 0.7), linewidth = 1)
    end
    lines!(ax, x, cdf.(Normal(), x); color = PALETTE.red, linestyle = :dash, linewidth = 2)
    xlims!(ax, -3, 3); ylims!(ax, 0, 1)
    push!(axs, ax)
end
linkyaxes!(axs...)
hideydecorations!(axs[2]; grid = false); hideydecorations!(axs[3]; grid = false)
save_fig(fig, @__DIR__, "dp-cdfs")

# ---- Figure 2
fig2 = Figure(size = (780, 320))
ax1 = Axis(fig2[1, 1]; xlabel = L"\text{index } k", ylabel = L"\text{weight } w_k", title = "Stick-breaking weights",
           xgridvisible = false)
rng = MersenneTwister(1994)
K = 25
for (k, (alpha, off)) in enumerate([(2, -0.2), (20, 0.2)])
    w, _ = stick_breaking(rng, alpha)
    barplot!(ax1, (1:K) .+ off, w[1:K]; width = 0.38, color = CYCLE[k], label = L"\alpha = %$alpha")
    lines!(ax1, 1:K, (1 / (1 + alpha)) .* (alpha / (1 + alpha)) .^ (0:K-1); color = CYCLE[k],
           linestyle = :dash, linewidth = 1.2)
end
xlims!(ax1, 0.3, K + 0.7); ylims!(ax1, 0, 0.6)
axislegend(ax1; position = :rt)

ax2 = Axis(fig2[1, 2]; xscale = log10, xlabel = L"\text{sample size } n \text{ (log scale)}",
           ylabel = L"\text{distinct values } K_n", title = "Number of distinct values")
rng = MersenneTwister(1953)
N = 5000; n = 1:N
for (k, alpha) in enumerate([1, 5, 25])
    for _ in 1:5
        new = rand(rng, N) .< alpha ./ (alpha .+ n .- 1)
        lines!(ax2, collect(n), cumsum(new); color = (CYCLE[k], 0.55), linewidth = 0.9)
    end
    lines!(ax2, collect(n), cumsum(alpha ./ (alpha .+ n .- 1)); color = CYCLE[k], linewidth = 2.2, label = L"\alpha = %$alpha")
end
xlims!(ax2, 1, N); ylims!(ax2, 0, 180)
axislegend(ax2; position = :lt)
save_fig(fig2, @__DIR__, "dp-weights-clusters")
