# Conjugate Bayesian inference for the normal model.
# (a) posterior of the mean with known variance; (b) Student-t marginal posterior under the
# normal-inverse-gamma prior versus a normal density with the same location and scale.
# Packages: CairoMakie, Distributions, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random

set_blog_theme!()
rng = MersenneTwister(2023)
mu_true, sigma = 1.0, 2.0
x = rand(rng, Normal(mu_true, sigma), 50)

m, tau = 0.0, 1.5
grid = range(-3, 4; length = 600)
fig = Figure(size = (760, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"\mu", ylabel = "density",
           title = L"\text{(a) known } \sigma^2 \text{: posterior of } \mu")
lines!(ax1, grid, pdf.(Normal(m, tau), grid); color = PALETTE.grey, linestyle = :dash, label = "prior")
for (n, col) in zip([2, 10, 50], [PALETTE.gold, PALETTE.red, PALETTE.blue])
    prec = 1 / tau^2 + n / sigma^2
    mn = (m / tau^2 + sum(x[1:n]) / sigma^2) / prec
    lines!(ax1, grid, pdf.(Normal(mn, 1 / sqrt(prec)), grid); color = col, label = "n = $n")
end
vlines!(ax1, [mu_true]; color = PALETTE.ink, linestyle = :dot, linewidth = 1)
axislegend(ax1; position = :lt)

m0, k0, a0, b0 = 0.0, 1.0, 1.0, 1.0
n = 5
xs = x[1:n]; xb = sum(xs) / n
kn = k0 + n; mn = (k0 * m0 + n * xb) / kn
an = a0 + n / 2
bn = b0 + 0.5 * sum((xs .- xb) .^ 2) + k0 * n * (xb - m0)^2 / (2 * kn)
sc = sqrt(bn / (an * kn))
g2 = range(mn - 8sc, mn + 8sc; length = 600)
tdens = pdf.(TDist(2an), (g2 .- mn) ./ sc) ./ sc
ndens = pdf.(Normal(mn, sc), g2)
keep = ndens .> 1e-5
ax2 = Axis(fig[1, 2]; xlabel = L"\mu", yscale = log10,
           title = L"\text{(b) unknown } \sigma^2,\ n = 5 \text{ (log scale)}")
lines!(ax2, g2, tdens; color = PALETTE.blue, label = "Student t, $(Int(2an)) d.f.")
lines!(ax2, g2[keep], ndens[keep]; color = PALETTE.red, linestyle = :dash, label = "normal, same scale")
ylims!(ax2, 1e-5, 2)
axislegend(ax2; position = :cb)
save_fig(fig, @__DIR__, "posteriors")
