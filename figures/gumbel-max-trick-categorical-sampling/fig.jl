# Gumbel-max sampling of a categorical distribution and the Gumbel-softmax relaxation.
# Packages: CairoMakie, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(7)
gumbel(rng, dims...) = -log.(-log.(rand(rng, dims...)))

# ---- Figure 1
a = [1.2, 0.3, -0.4, 0.9, -1.5, 0.0]; K = length(a)
pik = exp.(a) ./ sum(exp.(a)); logS = log(sum(exp.(a)))
N = 200_000
Y = gumbel(rng, N, K) .+ a'
I = [argmax(view(Y, i, :)) for i in 1:N]
M = [Y[i, I[i]] for i in 1:N]

fig = Figure(size = (780, 330))
ax = Axis(fig[1, 1]; xlabel = L"\text{category } k", ylabel = "probability", xticks = 1:K,
          title = L"2\times 10^5 \text{ draws, } K = 6")
freq = [count(==(k), I) / N for k in 1:K]
barplot!(ax, 1:K, pik; width = 0.6, color = PALETTE.light, strokecolor = PALETTE.grey, strokewidth = 1,
         label = L"\pi_k = e^{a_k} / \sum_j e^{a_j}")
scatter!(ax, 1:K, freq; color = PALETTE.blue, markersize = 11,
         label = L"\text{frequency of } \arg\max_j (a_j + G_j) = k")
ylims!(ax, 0, 0.46)
axislegend(ax; position = :rt, labelsize = 12)

ax = Axis(fig[1, 2]; xlabel = L"M = \max_j (a_j + G_j)", ylabel = "density",
          title = "The maximum is independent of the argmax")
x = range(-2, 8; length = 400)
for (j, col) in [(1, PALETTE.blue), (5, PALETTE.red)]
    stephist!(ax, M[I .== j]; bins = range(-2, 8; length = 70), normalization = :pdf, color = col,
              linewidth = 1.6, label = L"M \text{ given } I = %$j\ (\pi_{%$j} = %$(round(pik[j]; digits = 3)))")
end
lines!(ax, x, exp.(-(x .- logS) .- exp.(-(x .- logS))); color = PALETTE.ink, linestyle = :dash,
       linewidth = 1.4, label = L"\text{Gumbel}(\log \sum_j e^{a_j}) \text{ density}")
xlims!(ax, -2, 8); ylims!(ax, 0, 0.5)
axislegend(ax; position = :rt, labelsize = 12)
colsize!(fig.layout, 2, Relative(0.54))
save_fig(fig, @__DIR__, "argmax-and-max")

# ---- Figure 2: Gumbel-softmax samples on the simplex
a3 = log.([0.5, 0.3, 0.2])
V = [0.0 0.0; 1.0 0.0; 0.5 sqrt(3) / 2]
taus = [0.1, 0.5, 1.0, 5.0]; n = 1500
fig = Figure(size = (800, 240))
for (c, tau) in enumerate(taus)
    ax = Axis(fig[1, c]; title = L"\tau = %$tau", titlealign = :center, aspect = DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    Yt = (gumbel(rng, n, 3) .+ a3') ./ tau
    P = exp.(Yt .- maximum(Yt; dims = 2)); P ./= sum(P; dims = 2)
    xy = P * V
    win = [argmax(view(P, i, :)) for i in 1:n]
    lines!(ax, V[[1, 2, 3, 1], 1], V[[1, 2, 3, 1], 2]; color = PALETTE.grey, linewidth = 1)
    for j in 1:3
        sel = win .== j
        scatter!(ax, xy[sel, 1], xy[sel, 2]; color = (CYCLE[j], 0.5), markersize = 3)
    end
    for (j, (dx, dy, al)) in enumerate([(0.0, -0.08, :center), (0.0, -0.08, :center), (0.0, 0.06, :center)])
        text!(ax, V[j, 1] + dx, V[j, 2] + dy; text = L"e_{%$j}", align = (al, :center), fontsize = 14)
    end
    xlims!(ax, -0.08, 1.08); ylims!(ax, -0.16, 1.0)
end
save_fig(fig, @__DIR__, "gumbel-softmax")
