# Linear mixed model as a hierarchical model: partial pooling of group means by the BLUP in a
# random-intercept model, and the sampling distributions of the ML and REML estimators of the
# random-effect variance in a balanced one-way layout.
# Packages: CairoMakie, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1974)
mu, tau, sigma = 10.0, 1.0, 2.0
nj = [2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40]; J = length(nj)
b = tau .* randn(rng, J)
ybar = [mu + b[j] + mean(sigma .* randn(rng, nj[j])) for j in 1:J]
w = 1 ./ (tau^2 .+ sigma^2 ./ nj)
mu_hat = sum(w .* ybar) / sum(w)
lam = nj .* tau^2 ./ (nj .* tau^2 .+ sigma^2)
blup = mu_hat .+ lam .* (ybar .- mu_hat)

Jb, nb, tau2b, sig2b, reps = 6, 5, 1.0, 4.0, 20000
rng = MersenneTwister(1977)
reml = zeros(reps); ml = zeros(reps)
for r in 1:reps
    y = sqrt(tau2b) .* randn(rng, 1, Jb) .+ sqrt(sig2b) .* randn(rng, nb, Jb)
    gmeans = vec(mean(y; dims = 1)); gm = mean(y)
    msb = nb * sum((gmeans .- gm) .^ 2) / (Jb - 1)
    msw = sum((y .- gmeans') .^ 2) / (Jb * (nb - 1))
    reml[r] = max((msb - msw) / nb, 0.0)
    ml[r] = max(((1 - 1 / Jb) * msb - msw) / nb, 0.0)
end

fig = Figure(size = (790, 350))
ax1 = Axis(fig[1, 1]; xscale = log10, xticks = ([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"]),
           xlabel = L"\text{group size } n_j \text{ (log scale)}", ylabel = "estimate of group mean",
           title = "Partial pooling of group means")
hlines!(ax1, [mu_hat]; color = PALETTE.grey, linestyle = :dash, linewidth = 1)
arrows!(ax1, Float64.(nj), ybar, zeros(J), blup .- ybar; color = PALETTE.grey, linewidth = 1, arrowsize = 7)
scatter!(ax1, Float64.(nj), ybar; color = :white, strokecolor = PALETTE.ink, strokewidth = 1.2, markersize = 9,
         label = L"\text{group mean } \bar{y}_j")
scatter!(ax1, Float64.(nj), blup; color = PALETTE.blue, markersize = 9, label = L"\text{BLUP } \hat\mu + \hat{b}_j")
scatter!(ax1, Float64.(nj), mu .+ b; color = PALETTE.red, marker = :xcross, markersize = 10,
         label = L"\text{true } \mu + b_j")
xlims!(ax1, 1.7, 48)
axislegend(ax1; position = :rb, labelsize = 12)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{estimate of } \tau^2", ylabel = "distribution function",
           title = L"\text{ML and REML for } \tau^2,\ J = 6,\ n = 5")
grid = range(0, 6; length = 1201)
for (est, col, name) in [(reml, PALETTE.blue, "REML"), (ml, PALETTE.red, "ML")]
    s = sort(est)
    F = [searchsortedlast(s, g) / length(s) for g in grid]
    lines!(ax2, grid, F; color = col, linewidth = 2,
           label = @sprintf("%s: mean %.2f, P(0) = %.2f", name, mean(est), mean(est .== 0)))
    vlines!(ax2, [mean(est)]; color = col, linestyle = :dash, linewidth = 1)
end
vlines!(ax2, [tau2b]; color = PALETTE.ink, linewidth = 1)
text!(ax2, tau2b + 0.12, 0.93; text = L"\text{true } \tau^2 = 1", fontsize = 13)
xlims!(ax2, 0, 6); ylims!(ax2, 0, 1.02)
axislegend(ax2; position = :rb, labelsize = 12)
save_fig(fig, @__DIR__, "pooling-reml")
