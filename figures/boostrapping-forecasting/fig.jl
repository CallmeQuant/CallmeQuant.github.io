# Markov-chain bootstrap (Willemain, Smart and Schwarz) for intermittent demand on a simulated series
# (note: bootstrapping intermittent demand with a Markov chain).
# Packages: CairoMakie, Distributions, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(42)
p01, p11, size_mean = 0.10, 0.60, 3.0
n_hist, L, B = 730, 12, 20000

function simulate(n, state, rng)
    y = zeros(Int, n)
    for t in 1:n
        state = rand(rng) < (state == 1 ? p11 : p01) ? 1 : 0
        state == 1 && (y[t] = 1 + rand(rng, Poisson(size_mean)))
    end
    return y
end

y = simulate(n_hist, rand(rng) < p01 / (p01 + 1 - p11) ? 1 : 0, rng)
last = y[end] > 0 ? 1 : 0
truth = [sum(simulate(L, last, rng)) for _ in 1:B]

s = Int.(y .> 0)
nij = zeros(2, 2)
for t in 1:(n_hist - 1)
    nij[s[t] + 1, s[t + 1] + 1] += 1
end
P = nij ./ sum(nij; dims = 2)
sizes = y[y .> 0]
function wss(rng)
    state, total = last, 0
    for _ in 1:L
        state = rand(rng) < P[state + 1, 2] ? 1 : 0
        if state == 1
            x = sizes[rand(rng, 1:length(sizes))]
            xj = 1 + trunc(Int, x + sqrt(x) * randn(rng))
            total += xj > 0 ? xj : x
        end
    end
    return total
end
markov = [wss(rng) for _ in 1:B]
iid = [sum(y[rand(rng, 1:n_hist, L)]) for _ in 1:B]

fig = Figure(size = (840, 320))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{period } t", ylabel = L"\text{demand } y_t",
           title = "(a) last 120 periods of the history")
tt = collect((n_hist - 119):n_hist); yy = y[tt]
pos = yy .> 0
scatter!(ax1, tt[.!pos], zeros(count(.!pos)); marker = :vline, markersize = 5, color = PALETTE.grey)
for (ti, yi) in zip(tt[pos], yy[pos])
    lines!(ax1, [ti, ti], [0, yi]; color = PALETTE.blue, linewidth = 1.6)
end
scatter!(ax1, tt[pos], yy[pos]; color = PALETTE.blue, markersize = 6)
xlims!(ax1, tt[1] - 1, tt[end] + 1)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{total demand over } L = 12 \text{ periods}",
           ylabel = "distribution function", title = "(b) lead-time demand")
grid = 0:60
hlines!(ax2, [0.95]; color = PALETTE.grey, linewidth = 1, linestyle = :dot)
for (d, lab, col, ls) in [(truth, "true distribution", PALETTE.ink, :solid),
                          (markov, "Markov bootstrap", PALETTE.red, :solid),
                          (iid, "i.i.d. bootstrap", PALETTE.gold, :dash)]
    F = [mean(d .<= g) for g in grid]
    stairs!(ax2, collect(grid), F; step = :post, color = col, linestyle = ls, linewidth = 1.8, label = lab)
end
xlims!(ax2, 0, 60); ylims!(ax2, 0, 1.02)
axislegend(ax2; position = :rc)
colsize!(fig.layout, 1, Relative(0.53))
save_fig(fig, @__DIR__, "lead-time-demand")
