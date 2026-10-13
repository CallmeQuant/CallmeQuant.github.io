# Posterior consistency for a Bernoulli probability: contraction under a uniform prior, exponential
# decay of the posterior mass outside a neighbourhood, and concentration at the Kullback-Leibler
# projection when the prior support excludes the truth.
# Packages: CairoMakie, Distributions, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random

set_blog_theme!()
theta0, eps = 0.3, 0.05

# ---- Figure 1
rng = MersenneTwister(3)
S = cumsum(rand(rng, 10_000) .< theta0)
t = range(0, 1; length = 2001)
fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"\theta", ylabel = "posterior density", title = "Posterior along one sample path")
vlines!(ax1, [theta0]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
for (k, n) in enumerate([10, 100, 1000, 10000])
    lines!(ax1, t, pdf.(Beta(1 + S[n], 1 + n - S[n]), t); color = CYCLE[k], label = L"n = %$n")
end
xlims!(ax1, 0, 0.7); ylims!(ax1, 0, 95)
axislegend(ax1; position = :rt)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{sample size } n", ylabel = L"\Pi_n(|\theta - 0.3| > 0.05)",
           title = "Mass outside a neighbourhood", yscale = log10,
           yticks = (10.0 .^ (-10:2:0), [L"10^{%$k}" for k in -10:2:0]))
ns = 1:3000
for j in 1:8
    Sj = cumsum(rand(rng, length(ns)) .< theta0)
    out = [cdf(Beta(1 + s, 1 + n - s), theta0 - eps) + ccdf(Beta(1 + s, 1 + n - s), theta0 + eps) for (n, s) in zip(ns, Sj)]
    lines!(ax2, collect(ns), max.(out, 1e-300); color = (PALETTE.blue, 0.75), linewidth = 1)
end
xlims!(ax2, 0, 3000); ylims!(ax2, 1e-10, 3)
save_fig(fig, @__DIR__, "contraction")

# ---- Figure 2
kl(t0, t) = t0 * log(t0 / t) + (1 - t0) * log((1 - t0) / (1 - t))
tt = collect(range(0.001, 0.999; length = 2000))
in_supp = (tt .<= 0.2) .| (tt .>= 0.5)
rng = MersenneTwister(99)
S = cumsum(rand(rng, 400) .< theta0)

fig2 = Figure(size = (780, 330))
ax3 = Axis(fig2[1, 1]; xlabel = L"\theta", ylabel = L"K(\theta_0, \theta)", title = "Divergence from the truth",
           xgridvisible = false, ygridvisible = false)
vspan!(ax3, [0.0, 0.5], [0.2, 1.0]; color = PALETTE.light)
vlines!(ax3, [theta0]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
lines!(ax3, tt, kl.(theta0, tt); color = PALETTE.ink, linewidth = 2)
scatter!(ax3, [0.2, 0.5], kl.(theta0, [0.2, 0.5]); color = [PALETTE.red, PALETTE.grey], markersize = 11)
linesegments!(ax3, [0.2, 0.33], [kl(theta0, 0.2), 0.17]; color = PALETTE.ink, linewidth = 0.7)
text!(ax3, 0.33, 0.18; text = L"\theta^* = 0.2", align = (:left, :bottom), fontsize = 14)
text!(ax3, 0.31, 0.62; text = L"\theta_0", fontsize = 14)
text!(ax3, 0.64, 0.62; text = "prior support", align = (:center, :bottom), fontsize = 13)
xlims!(ax3, 0, 1); ylims!(ax3, 0, 0.7)

ax4 = Axis(fig2[1, 2]; xlabel = L"\theta", ylabel = "posterior density",
           title = L"\text{Posterior concentrates at } \theta^*")
vlines!(ax4, [theta0]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
for (k, n) in enumerate([25, 100, 400])
    s = S[n]
    lp = s .* log.(tt) .+ (n - s) .* log1p.(-tt)
    lp[.!in_supp] .= -Inf
    w = exp.(lp .- maximum(lp[in_supp]))
    w ./= sum(diff(tt) .* (w[1:end-1] .+ w[2:end]) ./ 2)
    lines!(ax4, tt, w; color = CYCLE[k], label = L"n = %$n")
end
xlims!(ax4, 0, 0.7); ylims!(ax4, 0, 165)
axislegend(ax4; position = :rt)
save_fig(fig2, @__DIR__, "misspecified-support")
