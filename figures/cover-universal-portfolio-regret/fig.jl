# Cover's universal portfolio on a simulated two-stock market.
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Printf

set_blog_theme!()

# Two stocks sharing a random-walk factor F, with a mean-reverting spread s.
rng = MersenneTwister(5)
n = 750
F = cumsum(vcat(0.0, 0.012 .* randn(rng, n)))
s = zeros(n + 1)
for t in 2:n+1
    s[t] = 0.9 * s[t-1] + 0.04 * randn(rng)
end
logA = F .+ s .+ 0.0002 .* (0:n); logB = F .- s
X = exp.(hcat(diff(logA), diff(logB)))

w = range(0, 1; length = 4001)
logS = vcat(zeros(1, length(w)), cumsum(log.(X[:, 1] .* w' .+ X[:, 2] .* (1 .- w')); dims = 1))
mx = maximum(logS; dims = 2)
W = exp.(logS .- mx)
w_hat = vec(sum(W .* w'; dims = 2) ./ sum(W; dims = 2))
log_up = vcat(0.0, cumsum(log.(w_hat[1:n] .* X[:, 1] .+ (1 .- w_hat[1:n]) .* X[:, 2])))
@assert maximum(abs.(log_up .- (log.(vec(mean(W; dims = 2))) .+ vec(mx)))) < 1e-6   # Lemma 1, numerically
k_star = argmax(logS[end, :]); w_star = w[k_star]
log_bh = log.(0.5 .* exp.(logA .- logA[1]) .+ 0.5 .* exp.(logB .- logB[1]))
tt = 0:n

fig = Figure(size = (780, 340))
ax = Axis(fig[1, 1]; xlabel = L"\text{period } t", ylabel = "log wealth", title = "Wealth over time (initial wealth 1)")
hlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.6)
lines!(ax, tt, logA .- logA[1]; color = PALETTE.grey, linewidth = 1, label = "stock A")
lines!(ax, tt, logB .- logB[1]; color = (PALETTE.violet, 0.8), linewidth = 1, label = "stock B")
lines!(ax, tt, log_bh; color = PALETTE.gold, linewidth = 1.4, label = "buy and hold, half in each")
lines!(ax, tt, logS[:, k_star]; color = PALETTE.green, linewidth = 1.7,
       label = @sprintf("best CRP in hindsight, w* = %.2f", w_star))
lines!(ax, tt, log_up; color = PALETTE.blue, linewidth = 2, label = "universal portfolio")
xlims!(ax, 0, n)
axislegend(ax; position = :lt, labelsize = 11)
ax = Axis(fig[1, 2]; xlabel = L"\text{weight } w \text{ on stock A}",
          ylabel = "final wealth after n = $n periods", title = "Every constant rebalancing rule")
lines!(ax, w, exp.(logS[end, :]); color = PALETTE.green, linewidth = 1.8,
       label = L"S_n(b)\text{, CRP with weight } w \text{ on A}")
hlines!(ax, [exp(log_up[end])]; color = PALETTE.blue, linewidth = 1.6, label = L"\hat S_n = \int_0^1 S_n(b)\,dw")
scatter!(ax, [0.0, 1.0], exp.(logS[end, [1, end]]); color = PALETTE.ink, markersize = 7)
text!(ax, 0.03, exp(logS[end, 1]) - 0.12; text = "all in B", fontsize = 12)
text!(ax, 0.97, exp(logS[end, end]) - 0.12; text = "all in A", fontsize = 12, align = (:right, :bottom))
axislegend(ax; position = :ct, labelsize = 11)
colsize!(fig.layout, 1, Relative(0.58))
save_fig(fig, @__DIR__, "wealth")

fig = Figure(size = (780, 320))
ax = Axis(fig[1, 1]; xlabel = L"\text{period } t", ylabel = "weight on stock A", title = "Portfolio weights")
lines!(ax, 1:n, [w[argmax(logS[t+1, :])] for t in 1:n]; color = PALETTE.green, linewidth = 1,
       label = L"\text{best CRP weight up to time } t")
lines!(ax, 1:n, w_hat[1:n]; color = PALETTE.blue, linewidth = 1.8, label = L"\text{universal portfolio weight } \hat w_t")
hlines!(ax, [w_star]; color = PALETTE.ink, linestyle = :dash, linewidth = 0.8)
xlims!(ax, 0, n); ylims!(ax, 0, 1)
axislegend(ax; position = :rt, labelsize = 11)
ax = Axis(fig[1, 2]; xlabel = L"\text{period } t", ylabel = "regret (log scale of wealth)", title = "Regret against the best CRP")
lines!(ax, tt, vec(mx) .- log_up; color = PALETTE.blue, linewidth = 1.8, label = L"\log S_t^* - \log \hat S_t")
lines!(ax, tt, log.(tt .+ 1) .+ 1; color = PALETTE.red, linestyle = :dash, linewidth = 1.5,
       label = L"\text{bound } (m-1)\log(t+1) + 1,\ m = 2")
xlims!(ax, 0, n); ylims!(ax, 0, 8.5)
axislegend(ax; position = :rc, labelsize = 11)
save_fig(fig, @__DIR__, "weights-regret")
