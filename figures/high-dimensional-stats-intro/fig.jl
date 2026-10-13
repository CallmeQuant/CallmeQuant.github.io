# (a) Tail bounds for a standard normal variable: Markov (applied to |X|), Chebyshev and Chernoff
# against the exact two-sided tail. (b) Johnson-Lindenstrauss: squared-distance ratios after a
# Gaussian random projection of N = 100 points from R^5000 to R^k.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random

set_blog_theme!()
rng = MersenneTwister(1916)
fig = Figure(size = (800, 350))

a = range(0.5, 5; length = 400)
ax1 = Axis(fig[1, 1]; xlabel = L"a", yscale = log10, title = L"\text{(a) tails of } X \sim \mathcal{N}(0,1)")
lines!(ax1, a, 2 .* ccdf.(Normal(), a); color = PALETTE.ink, label = L"\text{exact } \mathbb{P}(|X| \geq a)")
lines!(ax1, a, sqrt(2 / pi) ./ a; color = PALETTE.gold, linestyle = :dot, label = L"\text{Markov } \mathbb{E}|X|/a")
lines!(ax1, a, 1 ./ a .^ 2; color = PALETTE.red, linestyle = :dash, label = L"\text{Chebyshev } 1/a^2")
lines!(ax1, a, 2 .* exp.(-a .^ 2 ./ 2); color = PALETTE.blue, label = L"\text{Chernoff } 2e^{-a^2/2}")
ylims!(ax1, 1e-7, 2)
axislegend(ax1; position = :lb)

N, d = 100, 5000
P = randn(rng, N, d) .* rand(rng, Uniform(0.5, 2.0), N)
function sqdist(Y)
    G = Y * Y'; s = diag(G)
    D = s .+ s' .- 2 .* G
    return [D[i, j] for j in 1:size(Y, 1) for i in 1:j-1]
end
D0 = sqdist(P)
ax2 = Axis(fig[1, 2]; xlabel = L"\Vert f(x_i) - f(x_j) \Vert^2 / \Vert x_i - x_j \Vert^2", ylabel = "density",
           title = L"\text{(b) random projection},\ d = 5000,\ N = 100")
for (k, col) in zip([25, 100, 400], [PALETTE.gold, PALETTE.red, PALETTE.blue])
    A = randn(rng, k, d) ./ sqrt(k)
    ratio = sqdist(P * A') ./ D0
    hist!(ax2, ratio; bins = range(0.2, 2.0; length = 121), normalization = :pdf,
          color = (col, 0.3), strokecolor = col, strokewidth = 0.6, label = L"k = %$k")
end
vlines!(ax2, [1.0]; color = PALETTE.ink, linestyle = :dot, linewidth = 1)
axislegend(ax2; position = :rt)
save_fig(fig, @__DIR__, "tails-and-projection")
