# Generalized inverse of a distribution function with a flat stretch and a jump
# (note: inverse transform sampling). Mixture: 0.35 Unif(0,1) + 0.25 delta_{1.5} + 0.40 Unif(2,3).
# Packages: CairoMakie, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random

Fcdf(x) = x < 0 ? 0.0 : x < 1 ? 0.35x : x < 1.5 ? 0.35 : x < 2 ? 0.60 : x < 3 ? 0.60 + 0.40(x - 2) : 1.0
Finv(u) = u <= 0.35 ? u / 0.35 : u <= 0.60 ? 1.5 : 2 + (u - 0.60) / 0.40

set_blog_theme!()
blue, red = PALETTE.blue, PALETTE.red
fig = Figure(size = (860, 300))

ax1 = Axis(fig[1, 1]; xlabel = L"x", title = L"\text{(a) distribution function } F",
           yticks = ([0, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0], ["0", "0.2", "0.35", "0.5", "0.6", "0.8", "1"]))
for u in (0.2, 0.5, 0.8)
    x = Finv(u)
    lines!(ax1, [-0.4, x], [u, u]; color = red, linewidth = 1, linestyle = :dash)
    arrows!(ax1, [x], [u], [0.0], [-u + 0.02]; color = red, linewidth = 1, arrowsize = 8)
end
for (x0, x1, y0, y1) in [(-0.4, 0, 0, 0), (0, 1, 0, 0.35), (1, 1.5, 0.35, 0.35), (1.5, 2, 0.6, 0.6), (2, 3, 0.6, 1), (3, 3.4, 1, 1)]
    lines!(ax1, [x0, x1], [y0, y1]; color = blue)
end
scatter!(ax1, [1.5], [0.35]; color = :white, strokecolor = blue, strokewidth = 1.5, markersize = 9)
scatter!(ax1, [1.5], [0.6]; color = blue, markersize = 9)
xlims!(ax1, -0.4, 3.4); ylims!(ax1, 0, 1.05)

ax2 = Axis(fig[1, 2]; xlabel = L"u", title = L"\text{(b) quantile function } F^{-1}",
           xticks = ([0, 0.35, 0.6, 1.0], ["0", "0.35", "0.6", "1"]))
for (u0, u1) in [(0.0, 0.35), (0.35 + 1e-9, 0.6), (0.6 + 1e-9, 1.0)]
    uu = range(u0, u1; length = 50)
    lines!(ax2, collect(uu), Finv.(uu); color = blue)
end
scatter!(ax2, [0.35, 0.6], [1.0, 1.5]; color = blue, markersize = 9)
scatter!(ax2, [0.35, 0.6], [1.5, 2.0]; color = :white, strokecolor = blue, strokewidth = 1.5, markersize = 9)
xlims!(ax2, 0, 1); ylims!(ax2, -0.1, 3.2)

rng = MersenneTwister(2023)
X = sort(Finv.(rand(rng, 400)))
ax3 = Axis(fig[1, 3]; xlabel = L"x", title = L"\text{(c) draws } F^{-1}(U_i)")
xx = range(-0.4, 3.4; length = 1000)
lines!(ax3, collect(xx), Fcdf.(xx); color = (PALETTE.grey, 0.6), linewidth = 5, label = L"F")
stairs!(ax3, vcat(-0.4, X, 3.4), vcat(0.0, (1:length(X)) ./ length(X), 1.0); step = :post,
        color = red, linewidth = 1.2, label = L"\text{empirical}, n = 400")
xlims!(ax3, -0.4, 3.4); ylims!(ax3, 0, 1.05)
axislegend(ax3; position = :lt)
save_fig(fig, @__DIR__, "generalized-inverse")
