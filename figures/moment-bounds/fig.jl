# Upper-tail bounds for S - np, S ~ Binomial(n, p) with n = 200: exact tail versus Chebyshev,
# Hoeffding, Bernstein and the optimized Chernoff bound, for p = 0.5 and p = 0.05.
# Packages: CairoMakie, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions

set_blog_theme!()
n = 200
kl(q, p) = q * log(q / p) + (1 - q) * log((1 - q) / (1 - p))
fig = Figure(size = (800, 360))
axs = Axis[]
for (j, (p, tmax)) in enumerate([(0.5, 45), (0.05, 30)])
    t = range(0.5, tmax; length = 300)
    v = n * p * (1 - p); M = max(p, 1 - p)
    ax = Axis(fig[1, j]; yscale = log10, xlabel = L"\text{deviation } t",
              ylabel = j == 1 ? L"\mathbb{P}(S - np \geq t) \text{ and bounds}" : "",
              title = L"p = %$p,\ np(1-p) = %$(round(v; digits = 2))")
    exact = [ccdf(Binomial(n, p), ceil(n * p + ti) - 1) for ti in t]
    stairs!(ax, t, exact; color = PALETTE.ink, step = :post, label = "exact")
    lines!(ax, t, min.(1, v ./ t .^ 2); color = PALETTE.gold, linestyle = :dot, label = "Chebyshev")
    lines!(ax, t, exp.(-2 .* t .^ 2 ./ n); color = PALETTE.red, linestyle = :dash, label = "Hoeffding")
    lines!(ax, t, exp.(-t .^ 2 ./ (2v .+ 2M .* t ./ 3)); color = PALETTE.green, linestyle = :dashdot, label = "Bernstein")
    lines!(ax, t, exp.(-n .* kl.(p .+ t ./ n, p)); color = PALETTE.blue, label = "Chernoff (optimized)")
    ylims!(ax, 1e-12, 1.5)
    push!(axs, ax)
end
linkyaxes!(axs...)
axislegend(axs[2]; position = :lb)
save_fig(fig, @__DIR__, "binomial-tails")
