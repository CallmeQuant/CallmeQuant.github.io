# Normal scale mixtures: Laplace and Student t with unit variance, mixing laws, EM weights.
# Packages: CairoMakie, Distributions, Random, StatsBase
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, StatsBase

set_blog_theme!()
rng = MersenneTwister(1974)
nu = 5
b = 1 / sqrt(2)
s_t = sqrt((nu - 2) / nu)
M = 2_000_000
Z = randn(rng, M)
X_lap = sqrt.(rand(rng, Exponential(2b^2), M)) .* Z
X_t = sqrt.((nu - 2) ./ rand(rng, Chisq(nu), M)) .* Z
edges = range(-8, 8; length = 65)
mids = (edges[1:end-1] .+ edges[2:end]) ./ 2
w = step(edges)

fig = Figure(size = (1000, 330))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = L"x", title = "Densities, unit variance (log scale)")
x = range(-8, 8; length = 600)
lines!(ax1, x, pdf.(Normal(), x); color = PALETTE.grey, label = "normal")
lines!(ax1, x, pdf.(Laplace(0, b), x); color = PALETTE.blue, label = "Laplace")
lines!(ax1, x, pdf.(TDist(nu), x ./ s_t) ./ s_t; color = PALETTE.red, label = L"\text{Student } t_5")
for (X, col) in [(X_lap, PALETTE.blue), (X_t, PALETTE.red)]
    h = fit(Histogram, X, edges).weights
    ok = h .> 20
    scatter!(ax1, mids[ok], h[ok] ./ (M * w); color = :white, strokecolor = col, strokewidth = 1, markersize = 5)
end
xlims!(ax1, -8, 8); ylims!(ax1, 1e-6, 1.2)
axislegend(ax1; position = :rt, labelsize = 12)

ax2 = Axis(fig[1, 2]; xlabel = L"v", title = L"\text{Mixing densities of } V")
v = range(1e-3, 4; length = 500)
lines!(ax2, v, pdf.(Exponential(2b^2), v); color = PALETTE.blue, label = L"\text{Laplace: } V \sim \mathrm{Exp},\ \text{mean } 1")
lines!(ax2, v, pdf.(InverseGamma(nu / 2, (nu - 2) / 2), v); color = PALETTE.red,
       label = L"t_5\text{: } V = 3/G,\ G \sim \chi^2_5")
vlines!(ax2, [1.0]; color = PALETTE.grey, linewidth = 2.4, label = L"\text{normal: } V = 1")
xlims!(ax2, 0, 4); ylims!(ax2, 0, 1.6)
axislegend(ax2; position = :rt, labelsize = 12)

ax3 = Axis(fig[1, 3]; xlabel = L"\text{standardized residual } r", title = L"\text{EM weights } \mathbb{E}[V^{-1} \mid r]")
r = range(-6, 6; length = 801)
hlines!(ax3, [1.0]; color = PALETTE.grey, label = "normal")
lines!(ax3, r, (nu + 1) ./ (nu .+ r .^ 2); color = PALETTE.red, label = L"t_5")
lines!(ax3, r, 4 ./ (3 .+ r .^ 2); color = PALETTE.gold, label = L"t_3")
lines!(ax3, r, 2 ./ (1 .+ r .^ 2); color = PALETTE.green, label = "Cauchy")
rl = filter(z -> abs(z) > 0.15, collect(r))
lines!(ax3, rl[rl .< 0], 1 ./ abs.(rl[rl .< 0]); color = PALETTE.blue, label = "Laplace")
lines!(ax3, rl[rl .> 0], 1 ./ rl[rl .> 0]; color = PALETTE.blue)
xlims!(ax3, -6, 6); ylims!(ax3, 0, 2.6)
axislegend(ax3; position = :lt, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.37))
save_fig(fig, @__DIR__, "scale-mixtures")
