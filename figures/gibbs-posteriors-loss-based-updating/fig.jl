# Gibbs posteriors for the median under absolute-error loss with Gamma(2, 1) data: posteriors for
# several learning rates against a Gaussian-model posterior, and frequentist coverage of 95%
# credible intervals as a function of the learning rate.
# Packages: CairoMakie, Distributions, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()
shape = 2.0
med = quantile(Gamma(shape, 1.0), 0.5)
omega_star = 2 * pdf(Gamma(shape, 1.0), med)
grid = collect(range(0.0, 8.0; length = 3201)); dg = grid[2] - grid[1]

function gibbs(x, omegas)
    loss = vec(sum(abs.(x .- grid'); dims = 1))
    lw = -omegas * loss'
    w = exp.(lw .- maximum(lw; dims = 2))
    return w ./ (sum(w; dims = 2) .* dg)
end

rng = MersenneTwister(31)
n = 50
x = rand(rng, Gamma(shape, 1.0), n)
post = gibbs(x, [0.2, omega_star, 3.0])
ytop = 1.12 * maximum(post)

fig = Figure(size = (780, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"\theta", ylabel = "posterior density", title = L"\text{One sample},\ n = 50")
vlines!(ax1, [med]; color = PALETTE.ink, linestyle = :dot, linewidth = 1)
vlines!(ax1, [shape]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
text!(ax1, med - 0.03, 0.95 * ytop; text = "median", align = (:right, :center), fontsize = 12)
text!(ax1, shape + 0.03, 0.95 * ytop; text = "mean", align = (:left, :center), fontsize = 12, color = PALETTE.grey)
labels = [L"\omega = 0.2", L"\omega = \omega^* \approx %$(round(omega_star; digits = 2))", L"\omega = 3"]
for k in 1:3
    lines!(ax1, grid, post[k, :]; color = CYCLE[k], label = labels[k])
end
lines!(ax1, grid, pdf.(Normal(mean(x), std(x) / sqrt(n)), grid); color = PALETTE.grey, linestyle = :dash,
       linewidth = 1.8, label = "Gaussian model")
xlims!(ax1, 0, 3.4); ylims!(ax1, 0, ytop)
axislegend(ax1; position = (0.0, 0.85), labelsize = 12)

reps = 2000
omega_grid = exp.(range(log(0.05), log(5.0); length = 31))
ax2 = Axis(fig[1, 2]; xscale = log10, xlabel = L"\text{learning rate } \omega \text{ (log scale)}",
           ylabel = "coverage of 95% interval", title = "Frequentist coverage for the median",
           xticks = ([0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"]))
hlines!(ax2, [0.95]; color = PALETTE.grey, linestyle = :dash, linewidth = 1)
vlines!(ax2, [omega_star]; color = PALETTE.ink, linestyle = :dot, linewidth = 1)
text!(ax2, omega_star * 1.08, 0.08; text = L"\omega^*", fontsize = 14)
rng = MersenneTwister(32)
for (k, n) in enumerate([50, 400])
    hits = zeros(length(omega_grid))
    for _ in 1:reps
        p = gibbs(rand(rng, Gamma(shape, 1.0), n), omega_grid)
        c = cumsum(p; dims = 2) .* dg
        lo = [grid[findfirst(>=(0.025), c[i, :])] for i in axes(c, 1)]
        hi = [grid[findfirst(>=(0.975), c[i, :])] for i in axes(c, 1)]
        hits .+= (lo .<= med) .& (med .<= hi)
    end
    scatterlines!(ax2, omega_grid, hits ./ reps; color = CYCLE[k], markersize = 6, linewidth = 1.6, label = L"n = %$n")
end
og = exp.(range(log(0.05), log(5.0); length = 400))
lines!(ax2, og, 2 .* cdf.(Normal(), quantile(Normal(), 0.975) .* sqrt.(omega_star ./ og)) .- 1;
       color = PALETTE.ink, linewidth = 1.2, label = "asymptotic")
xlims!(ax2, 0.05, 5); ylims!(ax2, 0, 1.02)
axislegend(ax2; position = :lb)
save_fig(fig, @__DIR__, "gibbs-median")
