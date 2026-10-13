# Chi lengths of projections of a standard Gaussian vector; chi densities; t statistic as a cotangent.
# Packages: CairoMakie, Distributions, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, LinearAlgebra, Statistics

set_blog_theme!()
rng = MersenneTwister(1908)
n, k, M = 10, 3, 1500
Z = randn(rng, M, n)
Q = Matrix(qr(randn(rng, n, n)).Q)
Y = Z * Q
a = [norm(Y[i, 1:k]) for i in 1:M]
b = [norm(Y[i, k+1:n]) for i in 1:M]

fig = Figure(size = (1000, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{leg in } L\ (\chi_3)", ylabel = L"\text{leg in } L^\perp\ (\chi_7)",
           title = L"n = 10,\ \dim L = 3", aspect = DataAspect())
scatter!(ax1, a, b; markersize = 3, color = (PALETTE.grey, 0.5))
th = range(0, pi / 2; length = 200)
for (q, ls) in [(0.1, :dot), (0.5, :dash), (0.9, :dot)]
    r = quantile(Chi(n), q)
    lines!(ax1, r .* cos.(th), r .* sin.(th); color = PALETTE.ink, linestyle = ls, linewidth = 1)
end
i = 8
lines!(ax1, [0, a[i], a[i], 0], [0, 0, b[i], 0]; color = PALETTE.red, linewidth = 2)
text!(ax1, a[i] / 2, -0.12; text = L"\Vert PZ\Vert", color = PALETTE.red, align = (:center, :top), fontsize = 13)
text!(ax1, a[i] + 0.1, b[i] / 2; text = L"\Vert (I-P)Z\Vert", color = PALETTE.red, align = (:left, :center), fontsize = 13)
xlims!(ax1, 0, 5); ylims!(ax1, -0.55, 5.5)

ax2 = Axis(fig[1, 2]; xlabel = L"r", title = L"\text{Densities of } \chi_k \text{; dotted at } \sqrt{k}")
r = range(0, 9; length = 600)
for (j, kk) in enumerate([1, 2, 3, 10, 50])
    lines!(ax2, r, pdf.(Chi(kk), r); color = CYCLE[j], label = L"\chi_{%$kk}")
    vlines!(ax2, [sqrt(kk)]; color = CYCLE[j], linestyle = :dot, linewidth = 0.8)
end
xlims!(ax2, 0, 9); ylims!(ax2, 0, 0.9)
axislegend(ax2; position = :rt)

m, S = 6, 200_000
X = 2.0 .+ 3.0 .* randn(rng, S, m)
Zc = (X .- 2.0) ./ 3.0
signed_leg = sqrt(m) .* vec(mean(Zc; dims = 2))
other_leg = vec(sqrt.(sum((Zc .- mean(Zc; dims = 2)) .^ 2; dims = 2)))
theta = atan.(other_leg, signed_leg)
T = sqrt(m - 1) ./ tan.(theta)
ax3 = Axis(fig[1, 3]; xlabel = L"\sqrt{n-1}\,\cot\vartheta,\ n = 6", title = L"\text{The } t \text{ statistic as a cotangent}")
hist!(ax3, filter(z -> abs(z) < 6, T); bins = range(-6, 6; length = 81), normalization = :pdf,
      color = (PALETTE.grey, 0.45), strokecolor = :white, strokewidth = 0.3)
x = range(-6, 6; length = 400)
lines!(ax3, x, pdf.(TDist(m - 1), x); color = PALETTE.blue, label = L"t_5 \text{ density}")
xlims!(ax3, -6, 6); ylims!(ax3, 0, 0.45)
axislegend(ax3; position = :rt)
save_fig(fig, @__DIR__, "chi-triangles")
