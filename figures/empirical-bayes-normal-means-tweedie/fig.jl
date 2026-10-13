# Empirical Bayes for p = 1000 normal means with a two-component prior: the marginal density of X
# and its log-polynomial estimate, and the estimated rules (James-Stein and Tweedie) against the
# oracle posterior mean.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random, Statistics

set_blog_theme!()
p, w1, mu1, s1 = 1000, 0.3, 3.0, 0.5
phi(x) = pdf(Normal(), x)
function oracle(x)
    v = 1 + s1^2
    f0 = (1 - w1) * phi(x); f1 = w1 * pdf(Normal(mu1, sqrt(v)), x)
    return f1 * (mu1 + s1^2 / v * (x - mu1)) / (f0 + f1)
end
marginal(x) = (1 - w1) * phi(x) + w1 * pdf(Normal(mu1, sqrt(1 + s1^2)), x)

"Tweedie's formula with log f a degree-5 polynomial fitted to histogram counts (Poisson IRLS)."
function tweedie(x, at; degree = 5, nbins = 60)
    edges = range(minimum(x) - 1e-9, maximum(x) + 1e-9; length = nbins + 1)
    mids = (edges[1:end-1] .+ edges[2:end]) ./ 2; width = step(edges)
    y = [count(e -> edges[k] <= e < edges[k+1], x) for k in 1:nbins]
    c = mean(mids); s = sqrt(mean((mids .- c) .^ 2))
    Z = ((mids .- c) ./ s) .^ (0:degree)'
    beta = zeros(degree + 1); beta[1] = log(mean(y))
    for _ in 1:100
        mu = exp.(Z * beta)
        stp = (Z' * (mu .* Z)) \ (Z' * (y .- mu))
        beta .+= stp
        maximum(abs.(stp)) < 1e-10 && break
    end
    u = (at .- c) ./ s
    logf = (u .^ (0:degree)') * beta
    dlogf = (u .^ (0:degree-1)') * (beta[2:end] .* (1:degree)) ./ s
    return at .+ dlogf, exp.(logf) ./ (length(x) * width)
end

rng = MersenneTwister(1956)
z = rand(rng, p) .< w1
theta = ifelse.(z, mu1 .+ s1 .* randn(rng, p), 0.0)
x = theta .+ randn(rng, p)
grid = collect(range(-3.5, 6.5; length = 600))
tw, fhat = tweedie(x, grid)
xb = mean(x); b = min(1.0, (p - 3) / sum((x .- xb) .^ 2))

fig = Figure(size = (780, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"x", ylabel = "density", title = L"\text{Marginal density of } X,\ p = 1000",
           xgridvisible = false)
hist!(ax1, x; bins = 40, normalization = :pdf, color = PALETTE.light, strokecolor = :white, strokewidth = 0.5,
      label = L"\text{observed } X_i")
lines!(ax1, grid, marginal.(grid); color = PALETTE.ink, linewidth = 1.8, label = L"\text{true } f")
lines!(ax1, grid, fhat; color = PALETTE.red, linestyle = :dash, linewidth = 1.8,
       label = L"\text{fitted } \hat{f} \text{ (Poisson regression)}")
xlims!(ax1, -3.5, 6.5); ylims!(ax1, 0, 0.36)
axislegend(ax1; position = :rt, labelsize = 12)

inside = (grid .>= quantile(x, 0.005)) .& (grid .<= quantile(x, 0.995))
ax2 = Axis(fig[1, 2]; xlabel = L"x", ylabel = L"\text{estimate of } \theta", title = "Estimated posterior means")
lines!(ax2, grid, grid; color = PALETTE.grey, linestyle = :dot, linewidth = 1.2, label = L"\text{MLE } \hat\theta_i = X_i")
lines!(ax2, grid, xb .+ (1 - b) .* (grid .- xb); color = PALETTE.green, label = "James–Stein (linear)")
lines!(ax2, grid[inside], tw[inside]; color = PALETTE.red, linestyle = :dash, label = "Tweedie, estimated")
lines!(ax2, grid, oracle.(grid); color = PALETTE.ink, linewidth = 1.8, label = "oracle Bayes rule")
xlims!(ax2, -3.5, 6.5); ylims!(ax2, -1.5, 5.5)
axislegend(ax2; position = :lt, labelsize = 12)
save_fig(fig, @__DIR__, "eb-rules")
