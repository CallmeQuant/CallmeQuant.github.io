# Lindley's paradox: posterior probability of a point null at a fixed z-statistic, and the
# Bayesian critical value of |z| as functions of n (tau = sigma, pi0 = 1/2).
# Packages: CairoMakie
include(joinpath(@__DIR__, "..", "_style", "style.jl"))

set_blog_theme!()
n = 10 .^ range(0, 7; length = 600)
r = n
zs = [(1.96, "0.05"), (2.576, "0.01"), (3.291, "0.001")]
logticks = (collect(10.0 .^ (0:7)), [L"10^{%$k}" for k in 0:7])

fig = Figure(size = (780, 420))
ax1 = Axis(fig[1, 1]; xscale = log10, xticks = logticks, xlabel = L"\text{sample size } n",
           ylabel = L"P(H_0 \mid X)", title = L"\text{Posterior probability of } H_0 \text{ at fixed } z")
for (k, (z, p)) in enumerate(zs)
    B = sqrt.(1 .+ r) .* exp.(-0.5 * z^2 .* r ./ (1 .+ r))
    lines!(ax1, n, B ./ (1 .+ B); color = CYCLE[k], label = L"z = %$(round(z; digits = 2))\ (p = %$p)")
    hlines!(ax1, [1 / (1 + exp(z^2 / 2))]; color = CYCLE[k], linewidth = 1, linestyle = :dash)
end
xlims!(ax1, 1, 1e7); ylims!(ax1, 0, 1)

ax2 = Axis(fig[1, 2]; xscale = log10, xticks = logticks, xlabel = L"\text{sample size } n",
           ylabel = L"\text{critical value of } |z|", title = "Rejection thresholds")
c = sqrt.((1 .+ 1 ./ r) .* log1p.(r))
lines!(ax2, n, c; color = PALETTE.blue, label = L"\text{Bayes: } P(H_0 \mid X) = 1/2")
hlines!(ax2, [1.96]; color = PALETTE.red, linestyle = :dash, linewidth = 1.6, label = "5% test")
hlines!(ax2, [3.291]; color = PALETTE.gold, linestyle = :dash, linewidth = 1.6, label = "0.1% test")
xlims!(ax2, 1, 1e7); ylims!(ax2, 0, 4.6)

Legend(fig[2, 1], ax1; tellwidth = false, halign = :left)
Legend(fig[2, 2], ax2; tellwidth = false, halign = :left)
save_fig(fig, @__DIR__, "lindley")
