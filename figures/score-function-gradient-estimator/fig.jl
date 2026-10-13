# Variance of score-function and pathwise gradient estimators for d/dtheta E[X^2], X ~ N(theta, sigma^2).
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics

set_blog_theme!()
sigma = 1.0
rng = MersenneTwister(12)

function estimators(theta, z)
    x = theta .+ sigma .* z; score = z ./ sigma
    return Dict(:sf => x .^ 2 .* score,
                :mean => (x .^ 2 .- (theta^2 + sigma^2)) .* score,
                :opt => (x .^ 2 .- (theta^2 + 3sigma^2)) .* score,
                :path => 2 .* x)
end
exact = Dict(:sf => t -> t^4 / sigma^2 + 14t^2 + 15sigma^2, :mean => t -> 8t^2 + 10sigma^2,
             :opt => t -> 8t^2 + 6sigma^2, :path => t -> 4sigma^2)
keys_ = [:sf, :mean, :opt, :path]
names = Dict(:sf => L"\text{score function, no baseline}", :mean => L"\text{baseline } b = \mathbb{E} f(X)",
             :opt => L"\text{optimal baseline } b^*", :path => L"\text{pathwise (reparameterization)}")
cols = Dict(:sf => PALETTE.red, :mean => PALETTE.gold, :opt => PALETTE.green, :path => PALETTE.blue)

fig = Figure(size = (780, 340))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = L"\theta", ylabel = "variance of one-sample estimator",
          title = L"f(x) = x^2,\ X \sim N(\theta, 1)")
t = range(0, 4; length = 200); te = 0:0.5:4
z = randn(rng, 200_000)
for k in keys_
    lines!(ax, t, exact[k].(t); color = cols[k], label = names[k])
    scatter!(ax, te, [var(estimators(th, z)[k]) for th in te]; color = cols[k], markersize = 7)
end
xlims!(ax, 0, 4); ylims!(ax, 2, 2e3)
axislegend(ax; position = :lt, labelsize = 12)

theta, n = 2.0, 5000
z = randn(rng, n); est = estimators(theta, z); m = 1:n
ax = Axis(fig[1, 2]; xscale = log10, xlabel = L"\text{number of samples } n",
          ylabel = "running average of estimates",
          title = L"\text{Estimating } \nabla L(2) = 4 \text{, bands } \pm 2\,\mathrm{sd}/\sqrt{n}")
for k in [:sf, :opt, :path]
    half = 2 .* sqrt.(exact[k](theta) ./ m)
    band!(ax, m, 2theta .- half, 2theta .+ half; color = (cols[k], 0.12))
    lines!(ax, m, cumsum(est[k]) ./ m; color = cols[k], linewidth = 1.3)
end
hlines!(ax, [2theta]; color = PALETTE.ink, linestyle = :dash, linewidth = 0.9)
xlims!(ax, 5, n); ylims!(ax, -4, 12)
save_fig(fig, @__DIR__, "gradient-variance")
