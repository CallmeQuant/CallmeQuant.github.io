# Tweedie distributions with 1 < p < 2 as compound Poisson-gamma laws.
# Left: continuous part of the density (mu = 1, phi = 1). Right: sample variance against mean, log-log.
# Packages: CairoMakie, Distributions, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1984)

cpg(mu, phi, p) = (mu^(2 - p) / (phi * (2 - p)), (2 - p) / (p - 1), phi * (p - 1) * mu^(p - 1))   # (lambda, alpha, gamma)
function tw_density(y, mu, phi, p; nmax = 400)
    lam, alpha, gam = cpg(mu, phi, p)
    return sum(pdf(Poisson(lam), n) * pdf(Gamma(n * alpha, gam), y) for n in 1:nmax)
end
function rtweedie(size, mu, phi, p)
    lam, alpha, gam = cpg(mu, phi, p)
    N = rand(rng, Poisson(lam), size)
    return [n > 0 ? rand(rng, Gamma(n * alpha, gam)) : 0.0 for n in N]
end

fig = Figure(size = (790, 350))
ax1 = Axis(fig[1, 1]; title = L"\mu = 1,\ \phi = 1", xlabel = L"y", ylabel = "density of the continuous part")
y = range(0.005, 4; length = 800)
for (col, p) in zip(CYCLE, [1.1, 1.3, 1.5, 1.7, 1.9])
    p0 = exp(-cpg(1.0, 1.0, p)[1])
    p0s = p0 >= 0.01 ? @sprintf("%.2f", p0) : @sprintf("%.1e", p0)
    lines!(ax1, y, [tw_density(v, 1.0, 1.0, p) for v in y]; color = col, linewidth = 1.8,
           label = "p = $p:  P(Y = 0) = $p0s")
end
xlims!(ax1, 0, 4); ylims!(ax1, 0, 1.0)
axislegend(ax1; position = :rt, labelsize = 11)

ax2 = Axis(fig[1, 2]; title = L"\text{Sample variances, } n = 4000 \text{ per point}", xscale = log10, yscale = log10,
           xlabel = L"\text{mean } \mu \text{ (log scale)}", ylabel = "variance (log scale)")
mus = 10 .^ range(-1, 1; length = 9)
for (col, p) in zip([CYCLE[1], CYCLE[3], CYCLE[5]], [1.2, 1.5, 1.8])
    v = [var(rtweedie(4000, m, 1.0, p)) for m in mus]
    lines!(ax2, mus, mus .^ p; color = col, linewidth = 1.5, label = L"\mathrm{Var}\,Y = \mu^{%$p}")
    scatter!(ax2, mus, v; color = :white, strokecolor = col, strokewidth = 1.4, markersize = 9)
end
axislegend(ax2; position = :lt, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.53))
save_fig(fig, @__DIR__, "tweedie-cpg")
