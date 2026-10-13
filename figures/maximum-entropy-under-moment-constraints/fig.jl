# Maximum entropy pmfs on {0,...,10} with a mean constraint.
# Packages: CairoMakie, Distributions, Random, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Printf

set_blog_theme!()
rng = MersenneTwister(2026)
K = 10
k = collect(0:K)

function maxent(m)
    meanof(th) = (w = exp.(th .* k .- maximum(th .* k)); sum(w .* k) / sum(w))
    lo, hi = -60.0, 60.0                     # bisection: the mean is increasing in theta
    for _ in 1:200
        mid = (lo + hi) / 2
        meanof(mid) < m ? (lo = mid) : (hi = mid)
    end
    th = (lo + hi) / 2
    w = exp.(th .* k .- maximum(th .* k))
    return w ./ sum(w), th
end
H(p) = -sum(x > 0 ? x * log(x) : 0.0 for x in p)

fig = Figure(size = (800, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"k", ylabel = L"p_\theta(k)", xticks = 0:K,
           title = L"\text{Maximizers } p_\theta(k) \propto e^{\theta k}", xgridvisible = false)
means = [1.0, 3.0, 5.0, 7.5]
bw = 0.2
for (j, m) in enumerate(means)
    p, th = maxent(m)
    barplot!(ax1, k .+ (j - 2.5) * bw, p; width = bw, color = CYCLE[j],
             label = @sprintf("m = %g, θ = %.2f", m, th))
end
ylims!(ax1, 0, 0.68)
axislegend(ax1; position = :ct, nbanks = 2, labelsize = 11)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{mean } m", ylabel = "entropy (nats)", title = "Entropy against mean")
P = rand(rng, Dirichlet(ones(K + 1)), 6000)          # columns are pmfs
scatter!(ax2, vec(k' * P), [H(P[:, j]) for j in 1:size(P, 2)];
         markersize = 2, color = (PALETTE.grey, 0.35), label = "random pmfs (Dirichlet)")
ms = range(0.05, K - 0.05; length = 300)
lines!(ax2, ms, [H(maxent(m)[1]) for m in ms]; color = PALETTE.blue, label = "maximum entropy")
lines!(ax2, ms, [H(pdf.(Binomial(K, m / K), k)) for m in ms]; color = PALETTE.red,
       linestyle = :dash, linewidth = 1.6, label = "Binomial(10, m/10)")
hlines!(ax2, [log(K + 1)]; color = PALETTE.ink, linestyle = :dot, linewidth = 0.8)
text!(ax2, 0.2, log(K + 1) + 0.04; text = L"\log 11", fontsize = 12, align = (:left, :bottom))
xlims!(ax2, 0, K); ylims!(ax2, 0, 2.75)
axislegend(ax2; position = :cb, labelsize = 11)
save_fig(fig, @__DIR__, "maxent-discrete")
