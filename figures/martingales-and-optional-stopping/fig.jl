# Gambler's ruin paths and likelihood-ratio martingales with Ville's threshold.
# Packages: CairoMakie, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1939)
a, b, n_show = 6, 4, 14
function walk(rng)
    s, path = 0, [0]
    while -a < s < b
        s += rand(rng, Bool) ? 1 : -1
        push!(path, s)
    end
    return path
end

fig = Figure(size = (940, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"n", ylabel = L"S_{\tau \wedge n}", title = L"\text{Walks stopped at } -6 \text{ or } 4",
           yticks = -6:2:4)
hlines!(ax1, [b, -a]; color = PALETTE.ink, linewidth = 1.2)
for _ in 1:n_show
    p = walk(rng)
    col = p[end] == b ? PALETTE.blue : PALETTE.red
    stairs!(ax1, 0:length(p)-1, p; step = :post, color = (col, 0.75), linewidth = 1.2)
    scatter!(ax1, [length(p) - 1], [p[end]]; color = col, markersize = 7)
end
xlims!(ax1, 0, 80); ylims!(ax1, -a - 0.8, b + 0.8)

theta, N, lvl = 0.5, 400, 0.05
n = collect(1:N)
log10_lr(mu, m) = (theta .* cumsum(mu .+ randn(rng, m, N); dims = 2) .- n' .* theta^2 ./ 2) ./ log(10)
L0 = log10_lr(0.0, 20_000)
cross = mean(vec(maximum(L0; dims = 2)) .>= log10(1 / lvl))
L1 = log10_lr(theta, 8)
ax2 = Axis(fig[1, 2]; xlabel = L"n", ylabel = L"\log_{10} L_n",
           title = @sprintf("Fraction of null paths ever above 20: %.3f", cross))
for i in 1:40
    lines!(ax2, n, L0[i, :]; color = (PALETTE.grey, 0.8), linewidth = 0.8)
end
for i in 1:8
    lines!(ax2, n, L1[i, :]; color = PALETTE.green, linewidth = 1)
end
hlines!(ax2, [log10(1 / lvl)]; color = PALETTE.red, linestyle = :dash, linewidth = 1.4)
text!(ax2, 395, log10(1 / lvl) + 0.25; text = L"L_n = 1/\alpha = 20", color = PALETTE.red, align = (:right, :bottom))
xlims!(ax2, 0, N); ylims!(ax2, -8, 8)
Legend(fig[1, 2], [LineElement(color = PALETTE.grey), LineElement(color = PALETTE.green)],
       [L"\text{data from the null } N(0,1)", L"\text{data from } N(0.5,1)"];
       tellwidth = false, tellheight = false, halign = :right, valign = :top, margin = (8, 8, 8, 8))
save_fig(fig, @__DIR__, "martingale-paths")
