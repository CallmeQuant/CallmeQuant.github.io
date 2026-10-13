# Control variates for E[log(1 + exp(Z))], Z ~ N(0.5, 1.5^2).
# Packages: CairoMakie, FastGaussQuadrature, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using FastGaussQuadrature, LinearAlgebra, Random, Statistics, Printf

set_blog_theme!()
mu, s = 0.5, 1.5
f(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))          # softplus

# Reference values by Gauss-Hermite quadrature: E h(Z) = sum w_i h(mu + sqrt(2) s x_i) / sqrt(pi).
xq, wq = gausshermite(120)
zq = mu .+ sqrt(2) * s .* xq; wq = wq ./ sqrt(pi)
theta = sum(wq .* f.(zq))
var_f = sum(wq .* (f.(zq) .- theta) .^ 2)
controls(z, k) = hcat([z .- mu, (z .- mu) .^ 2 .- s^2][1:k]...)

function resid_var(k)
    k == 0 && return var_f
    G = controls(zq, k); F = f.(zq) .- theta
    Sgg = G' * (wq .* G); Sgf = G' * (wq .* F)
    return var_f - dot(Sgf, Sgg \ Sgf)
end

function estimate(z, k)
    y = f.(z)
    k == 0 && return mean(y)
    X = hcat(ones(length(z)), controls(z, k))
    return (X \ y)[1]                                        # least-squares intercept
end

fig = Figure(size = (780, 350))

# (a) the regression picture for one sample
rng = MersenneTwister(5)
z = mu .+ s .* randn(rng, 60)
g, y = z .- mu, f.(z)
a, b = hcat(ones(60), g) \ y
ax = Axis(fig[1, 1]; title = L"\text{One sample, } n = 60", xlabel = L"\text{control } g = Z - \mu",
          ylabel = L"f(Z) = \log(1 + e^{Z})")
hlines!(ax, [theta]; color = PALETTE.ink, linestyle = :dot, linewidth = 0.8)
vlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.6)
scatter!(ax, g, y; color = (PALETTE.grey, 0.8), markersize = 6)
lines!(ax, [-4, 4], a .+ b .* [-4, 4]; color = PALETTE.blue, linewidth = 1.8)
scatter!(ax, [mean(g)], [mean(y)]; color = PALETTE.red, markersize = 11,
         label = L"\text{plain mean } \bar f = %$(round(mean(y); digits = 3))")
scatter!(ax, [0.0], [a]; color = PALETTE.blue, marker = :rect, markersize = 11,
         label = L"\text{line at } g = 0: %$(round(a; digits = 3))")
text!(ax, 3.9, theta - 0.5; text = L"\theta = %$(round(theta; digits = 3))", align = (:right, :bottom))
xlims!(ax, -4, 4); ylims!(ax, -0.3, 4.6)
axislegend(ax; position = :lt)

# (b) root mean squared error against n
ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]; R = 2000
rng = MersenneTwister(6)
labels = [L"\text{plain Monte Carlo}", L"\text{control } Z - \mu",
          L"\text{controls } Z - \mu,\ (Z - \mu)^2 - \sigma^2"]
cols = [PALETTE.red, PALETTE.blue, PALETTE.green]
ax = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{sample size } n",
          ylabel = "root mean squared error",
          title = L"\text{Error against } n \text{ (lines: } \sqrt{\sigma_f^2 (1 - R^2)/n}\text{)}")
nn = exp.(range(log(8), log(7000); length = 50))
for k in 0:2
    rmse = [sqrt(mean((estimate(mu .+ s .* randn(rng, n), k) - theta)^2 for _ in 1:R)) for n in ns]
    lines!(ax, nn, sqrt.(resid_var(k) ./ nn); color = cols[k + 1], label = labels[k + 1])
    scatter!(ax, ns, rmse; color = cols[k + 1], markersize = 8)
end
ylims!(ax, 3e-4, 1.2)
axislegend(ax; position = :lb, labelsize = 12)
colsize!(fig.layout, 2, Relative(0.53))
save_fig(fig, @__DIR__, "control-variates")
