# James-Stein estimation. Figure 1: exact risk relative to the MLE. Figure 2: one draw with p = 30.
# Packages: CairoMakie, Distributions, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1961)

function js_risk(p, lam)
    k = 0:1999
    return p - (p - 2)^2 * sum(pdf.(Poisson(lam / 2), k) ./ (p - 2 .+ 2 .* k))
end

# ---- Figure 1 ----
r = range(0, 12; length = 241)
fig = Figure(size = (660, 350))
ax = Axis(fig[1, 1]; xlabel = L"\Vert \theta \Vert", ylabel = L"\text{risk relative to the MLE, } R(\theta)/p")
hlines!(ax, [1.0]; color = PALETTE.ink, linestyle = :dash, linewidth = 1)
text!(ax, 11.9, 1.015; text = "MLE", align = (:right, :bottom), fontsize = 13)
for (col, p) in zip(CYCLE, [3, 5, 10, 30])
    lines!(ax, r, [js_risk(p, t^2) / p for t in r]; color = col, label = L"p = %$p")
end
p, R = 10, 40000
rs = 0:1.0:12
pp = Float64[]
for t in rs
    theta = zeros(p); theta[1] = t
    X = theta' .+ randn(rng, R, p)
    shrink = max.(0, 1 .- (p - 2) ./ vec(sum(abs2, X; dims = 2)))
    push!(pp, mean(vec(sum(abs2, shrink .* X .- theta'; dims = 2))) / p)
end
scatter!(ax, collect(rs), pp; color = :white, strokecolor = CYCLE[3], strokewidth = 1.4, markersize = 8,
         label = L"\text{positive part, } p = 10 \text{ (simulated)}")
xlims!(ax, 0, 12); ylims!(ax, 0, 1.08)
axislegend(ax; position = :rb, labelsize = 12)
save_fig(fig, @__DIR__, "js-risk")

# ---- Figure 2 ----
p = 30
theta = 2.0 .+ 0.8 .* randn(rng, p)
X = theta .+ randn(rng, p)
js0 = (1 - (p - 2) / sum(abs2, X)) .* X
xbar = mean(X); jsm = xbar .+ (1 - (p - 3) / sum(abs2, X .- xbar)) .* (X .- xbar)
loss(e) = sum(abs2, e .- theta)
fig2 = Figure(size = (660, 370))
ax2 = Axis(fig2[1, 1]; title = L"\text{One draw, } p = 30;\ \text{dotted line: } \bar X", xlabel = L"\text{true mean } \theta_i",
           ylabel = "estimate")
lo = min(minimum(X), minimum(theta)) - 0.3; hi = max(maximum(X), maximum(theta)) + 0.3
lines!(ax2, [lo, hi], [lo, hi]; color = PALETTE.grey, linewidth = 1)
hlines!(ax2, [xbar]; color = PALETTE.blue, linestyle = :dot, linewidth = 1)
linesegments!(ax2, vec(permutedims(hcat(Point2f.(theta, X), Point2f.(theta, jsm)))); color = PALETTE.light, linewidth = 1.2)
scatter!(ax2, theta, X; color = PALETTE.grey, markersize = 8, label = @sprintf("MLE: loss %.1f", loss(X)))
scatter!(ax2, theta, js0; marker = :utriangle, color = :white, strokecolor = PALETTE.gold, strokewidth = 1.2,
         markersize = 9, label = @sprintf("James–Stein toward 0: loss %.1f", loss(js0)))
scatter!(ax2, theta, jsm; marker = :rect, color = PALETTE.blue, markersize = 8,
         label = @sprintf("James–Stein toward mean: loss %.1f", loss(jsm)))
xlims!(ax2, minimum(theta) - 0.3, maximum(theta) + 0.3); ylims!(ax2, lo, hi)
axislegend(ax2; position = :lt, labelsize = 12)
save_fig(fig2, @__DIR__, "js-one-draw")
