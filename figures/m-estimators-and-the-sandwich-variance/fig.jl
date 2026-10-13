# Huber location M-estimators: score functions and asymptotic variance versus threshold k.
# Packages: CairoMakie, Distributions, QuadGK, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, QuadGK, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(20261012)
const EPS, TAU = 0.1, 5.0
contam = MixtureModel([Normal(0, 1), Normal(0, TAU)], [1 - EPS, EPS])
laws = [(Normal(0, 1), L"N(0,1)", PALETTE.blue),
        (contam, L"0.9\,N(0,1) + 0.1\,N(0,25)", PALETTE.red),
        (TDist(3), L"\text{Student } t_3", PALETTE.green)]

function avar(d, k)
    B = 2 * quadgk(x -> x^2 * pdf(d, x), 0, k)[1] + 2k^2 * (1 - cdf(d, k))
    V = 2cdf(d, k) - 1
    return B / V^2
end

# Solve sum_i clamp(x_i - theta, -k, k) = 0 by bisection (the map is monotone in theta).
function huber_location(x, k; iters = 60)
    lo, hi = minimum(x), maximum(x)
    for _ in 1:iters
        mid = (lo + hi) / 2
        s = sum(clamp.(x .- mid, -k, k))
        s > 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

ks = range(0.05, 4.0; length = 160)
k_sim = [0.25, 0.75, 1.345, 2.0, 3.0]
n, R = 200, 4000

fig = Figure(size = (760, 340))
ax1 = Axis(fig[1, 1]; title = "Score functions", xlabel = L"\text{residual } u = x - \theta",
           ylabel = L"\varphi(u)")
u = range(-3, 3; length = 601)
lines!(ax1, u, u; color = PALETTE.grey, label = L"\text{mean: } \varphi(u) = u")
lines!(ax1, u, clamp.(u, -1.345, 1.345); color = PALETTE.blue, label = L"\text{Huber, } k = 1.345")
lines!(ax1, [-3, 0], [-1, -1]; color = PALETTE.gold, label = L"\text{median: } \mathrm{sign}(u)")
lines!(ax1, [0, 3], [1, 1]; color = PALETTE.gold)
xlims!(ax1, -3, 3); ylims!(ax1, -3, 3)
axislegend(ax1; position = :lt, labelsize = 12)

ax2 = Axis(fig[1, 2]; title = L"\text{Asymptotic variance of } \sqrt{n}(\hat\theta_n - \theta_0)",
           xlabel = L"\text{threshold } k", ylabel = L"\sigma^2(k) = B/V^2")
vlines!(ax2, [1.345]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
for (d, lab, col) in laws
    lines!(ax2, ks, [avar(d, k) for k in ks]; color = col, label = lab)
    est = [n * var([huber_location(rand(rng, d, n), k) for _ in 1:R]) for k in k_sim]
    scatter!(ax2, k_sim, est; color = :white, strokecolor = col, strokewidth = 1.5, markersize = 8)
end
xlims!(ax2, 0, 4.05); ylims!(ax2, 0.9, 3.5)
axislegend(ax2; position = :rt, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.42))
save_fig(fig, @__DIR__, "huber-variance")
