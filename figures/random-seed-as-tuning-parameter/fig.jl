# Seed-to-seed variability: random initialization in a nonconvex fit, random train/test splits,
# and the optimism of reporting the best of k seeds.
# Packages: CairoMakie, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random, Statistics, Printf

set_blog_theme!()

# ---- Figure 1: frequency estimation by gradient descent from random starting values
rng = MersenneTwister(101)
n = 100
x = 10 .* rand(rng, n)
y = sin.(1.3 .* x) .+ 0.3 .* randn(rng, n)
loss(w) = mean((y .- sin.(w .* x)) .^ 2)
grad(w) = -2 * mean((y .- sin.(w .* x)) .* cos.(w .* x) .* x)

w0 = 0.2 .+ 2.8 .* rand(MersenneTwister(7), 300)
w = copy(w0)
for _ in 1:3000
    w .-= 0.004 .* grad.(w)
end
ws = range(0.05, 3.2; length = 2000)
best = ws[argmin(loss.(ws))]
hit = abs.(w .- best) .< 0.02
vals = unique(round.(w; digits = 2)); counts = [count(==(v), round.(w; digits = 2)) for v in vals]

fig = Figure(size = (780, 330))
ax = Axis(fig[1, 1]; xlabel = L"\text{frequency } \omega", ylabel = L"\text{least-squares criterion } L_n(\omega)",
          title = "Nonconvex least squares, 300 random starts")
lines!(ax, ws, loss.(ws); color = PALETTE.ink, linewidth = 1.6)
scatter!(ax, vals, loss.(vals); markersize = 6 .+ 2 .* sqrt.(counts), color = (PALETTE.red, 0.8),
         label = "end points of gradient descent")
xlims!(ax, 0.05, 3.2); ylims!(ax, 0, 1.25)
axislegend(ax; position = (0.98, 0.45), labelsize = 11)
ax = Axis(fig[1, 2]; xlabel = L"\text{starting value } \omega^{(0)} \text{ (set by the seed)}",
          ylabel = L"\text{final value } \hat\omega", title = "Output as a function of the seed")
scatter!(ax, w0[hit], w[hit]; markersize = 5, color = PALETTE.blue,
         label = @sprintf("reaches global minimum (%.0f%%)", 100mean(hit)))
scatter!(ax, w0[.!hit], w[.!hit]; markersize = 5, color = PALETTE.red, label = "stops at a local minimum")
xlims!(ax, 0.1, 3.1); ylims!(ax, 0, 3.3)
axislegend(ax; position = :lt, labelsize = 11)
colsize!(fig.layout, 1, Relative(0.57))
save_fig(fig, @__DIR__, "nonconvex-starts")

# ---- Figure 2: test R^2 across random splits, and best-of-k optimism
rng = MersenneTwister(202)
n, p = 60, 8
X = randn(rng, n, p)
beta = [1.0, -0.8, 0.6, 0.5, 0.0, 0.0, 0.3, 0.0]
yv = X * beta .+ 1.2 .* randn(rng, n)
function test_r2(seed)
    idx = randperm(MersenneTwister(seed), n); tr, te = idx[1:45], idx[46:end]
    coef = hcat(ones(45), X[tr, :]) \ yv[tr]
    pred = hcat(ones(n - 45), X[te, :]) * coef
    return 1 - sum((yv[te] .- pred) .^ 2) / sum((yv[te] .- mean(yv[te])) .^ 2)
end
r2 = [test_r2(s) for s in 0:4999]
mu, sdv = mean(r2), std(r2)

fig = Figure(size = (780, 330))
ax = Axis(fig[1, 1]; xlabel = L"\text{test } R^2 \text{ (45 training, 15 test points)}", ylabel = "density",
          title = @sprintf("One data set, 5000 split seeds (sd %.2f)", sdv))
hist!(ax, r2; bins = 60, normalization = :pdf, color = PALETTE.light, strokecolor = :white, strokewidth = 0.5)
vlines!(ax, [mu]; color = PALETTE.blue, linewidth = 1.6, label = @sprintf("mean over seeds %.2f", mu))
vlines!(ax, [test_r2(42)]; color = PALETTE.red, linewidth = 1.6, linestyle = :dash,
        label = @sprintf("seed 42: %.2f", test_r2(42)))
xlims!(ax, -0.6, 1.0)
axislegend(ax; position = :lt, labelsize = 11)

ks = unique(round.(Int, exp.(range(log(1), log(200); length = 25))))
rb = MersenneTwister(303)
emax = [mean(maximum(rand(rb, r2, k)) for _ in 1:4000) for k in ks]
ax = Axis(fig[1, 2]; xscale = log10, xlabel = L"\text{number of seeds tried, } k",
          ylabel = L"\text{reported test } R^2", title = "Optimism from choosing the seed")
hlines!(ax, [1.0]; color = PALETTE.ink, linestyle = :dot, linewidth = 0.8)
text!(ax, 1.05, 1.02; text = L"R^2 \leq 1", fontsize = 12, align = (:left, :bottom))
scatterlines!(ax, ks, emax; color = PALETTE.red, markersize = 6, label = L"\text{expected best of } k \text{ seeds}")
lines!(ax, ks, mu .+ sdv .* sqrt.(2 .* log.(ks)); color = PALETTE.grey, linestyle = :dash,
       label = L"\mu + \sigma\sqrt{2\log k}")
hlines!(ax, [mu]; color = PALETTE.blue, linewidth = 1.6, label = L"\mu\text{: what a new split delivers}")
ylims!(ax, 0.2, 1.3)
axislegend(ax; position = :rb, labelsize = 11)
save_fig(fig, @__DIR__, "split-seeds")
