# Three copulas with Kendall's tau = 0.5, on the unit square and with fixed margins.
# Packages: CairoMakie, Distributions, Random, Statistics, StatsBase, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, StatsBase, Printf

set_blog_theme!()
rng = MersenneTwister(1959)
n, tau = 1500, 0.5
rho = sin(pi * tau / 2)            # Gaussian copula: tau = (2/pi) asin(rho)
theta = 2tau / (1 - tau)           # Clayton: tau = theta / (theta + 2)

Z = rand(rng, MvNormal([0.0, 0.0], [1.0 rho; rho 1.0]), n)'   # n x 2
U_gauss = cdf.(Normal(), Z)
function clayton(rng, n)           # Marshall-Olkin sampler
    V = rand(rng, Gamma(1 / theta, 1.0), n)
    E = rand(rng, Exponential(1.0), n, 2)
    return (1 .+ E ./ V) .^ (-1 / theta)
end
cases = [("Gaussian copula", U_gauss, PALETTE.blue),
         ("Clayton copula", clayton(rng, n), PALETTE.red),
         ("Survival Clayton copula", 1 .- clayton(rng, n), PALETTE.green)]

fig = Figure(size = (860, 580))
for (j, (name, U, col)) in enumerate(cases)
    ax = Axis(fig[1, j]; title = name, xlabel = L"u", ylabel = j == 1 ? L"v" : "",
              aspect = DataAspect(), xticks = [0, 0.5, 1], yticks = [0, 0.5, 1])
    scatter!(ax, U[:, 1], U[:, 2]; markersize = 2.5, color = (col, 0.6))
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    text!(ax, 0.03, 0.97; text = @sprintf("τ̂ = %.2f", corkendall(U[:, 1], U[:, 2])),
          align = (:left, :top), fontsize = 12, space = :relative)

    X = quantile.(Exponential(1.0), U[:, 1]); Y = quantile.(Normal(), U[:, 2])
    bx = Axis(fig[2, j]; xlabel = "x (Exp(1) margin)", ylabel = j == 1 ? "y (N(0,1) margin)" : "")
    scatter!(bx, X, Y; markersize = 2.5, color = (col, 0.6))
    xlims!(bx, 0, 6.5); ylims!(bx, -3.6, 3.6)
    text!(bx, 0.97, 0.04; text = @sprintf("Pearson r = %.2f", cor(X, Y)),
          align = (:right, :bottom), fontsize = 12, space = :relative)
end
save_fig(fig, @__DIR__, "three-copulas")
