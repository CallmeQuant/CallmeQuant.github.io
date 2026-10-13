# Riemann sums for int W dW with different evaluation points; geometric Brownian motion.
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1944)

# ---------- Figure 1 ----------
L = 18; K = 2^L
W = vcat(0.0, cumsum(sqrt(1 / K) .* randn(rng, K)))
levels = 1:(L - 1)
left = Float64[]; mid = Float64[]; right = Float64[]
for l in levels
    step = 2^(L - l)
    idx = 1:step:(K + 1)
    Wg = W[idx]; dW = diff(Wg); Wm = W[idx[1:end-1] .+ step ÷ 2]
    push!(left, sum(Wg[1:end-1] .* dW)); push!(mid, sum(Wm .* dW)); push!(right, sum(Wg[2:end] .* dW))
end
w1 = W[end]
n = 2.0 .^ levels
fig = Figure(size = (880, 340))
ax1 = Axis(fig[1, 1]; xscale = log2, xlabel = L"\text{number of intervals } n",
           title = L"\text{Sums } \sum_i W_{s_i}(W_{t_i} - W_{t_{i-1}}) \text{, one path}")
spec = [(right, L"\text{right endpoint} \to \frac{1}{2}(W_1^2 + 1)", PALETTE.red, 0.5 * (w1^2 + 1)),
        (mid, L"\text{midpoint} \to \frac{1}{2}W_1^2", PALETTE.green, 0.5 * w1^2),
        (left, L"\text{left endpoint} \to \frac{1}{2}(W_1^2 - 1)", PALETTE.blue, 0.5 * (w1^2 - 1))]
for (vals, lab, col, lim) in spec
    hlines!(ax1, [lim]; color = col, linestyle = :dash, linewidth = 1)
    scatterlines!(ax1, n, vals; color = col, markersize = 6, linewidth = 1.2, label = lab)
end
ylims!(ax1, min(spec[3][4], minimum(left)) - 0.35, max(spec[1][4], maximum(right)) + 0.95)
axislegend(ax1; position = :ct, labelsize = 12)

M, nint = 3000, 500
dWs = sqrt(1 / nint) .* randn(rng, M, nint)
Ws = cumsum(dWs; dims = 2)
lefts = vec(sum(hcat(zeros(M), Ws[:, 1:end-1]) .* dWs; dims = 2))
ax2 = Axis(fig[1, 2]; xlabel = L"W_1", title = "Left sums across 3,000 paths")
scatter!(ax2, Ws[:, end], lefts; markersize = 3, color = (PALETTE.blue, 0.35), label = L"\text{left sums}, n = 500")
x = range(-3.5, 3.5; length = 300)
lines!(ax2, x, 0.5 .* x .^ 2; color = PALETTE.grey, label = L"\frac{1}{2}w^2 \text{ (ordinary calculus)}")
lines!(ax2, x, 0.5 .* (x .^ 2 .- 1); color = PALETTE.ink, linestyle = :dash, label = L"\frac{1}{2}(w^2 - 1)")
xlims!(ax2, -3.5, 3.5); ylims!(ax2, -1, 5.5)
axislegend(ax2; position = :ct, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.54))
save_fig(fig, @__DIR__, "riemann-sums")

# ---------- Figure 2 ----------
mu, sigma, T, N, P = 0.1, 0.6, 10.0, 1000, 2000
t = collect(range(0, T; length = N + 1))
Wp = hcat(zeros(P), cumsum(sqrt(T / N) .* randn(rng, P, N); dims = 2))
S = exp.((mu - sigma^2 / 2) .* t' .+ sigma .* Wp)
fig = Figure(size = (680, 370))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = L"t", ylabel = L"S_t \text{ (log scale)}",
          title = L"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,\ \mu = 0.1,\ \sigma = 0.6,\ S_0 = 1")
for i in 1:25
    lines!(ax, t, S[i, :]; color = (PALETTE.grey, 0.7), linewidth = 0.6)
end
q10 = [quantile(S[:, j], 0.1) for j in 1:N+1]; q90 = [quantile(S[:, j], 0.9) for j in 1:N+1]
lines!(ax, t, q10; color = PALETTE.violet, linestyle = :dot, linewidth = 1.2)
lines!(ax, t, q90; color = PALETTE.violet, linestyle = :dot, linewidth = 1.2, label = "10% and 90% quantiles")
lines!(ax, t, exp.(mu .* t); color = PALETTE.red, linewidth = 2.6, label = L"\text{mean } e^{\mu t}")
lines!(ax, t, exp.((mu - sigma^2 / 2) .* t); color = PALETTE.blue, linewidth = 2.6,
       label = L"\text{median } e^{(\mu - \sigma^2/2)t}")
lines!(ax, t, vec(mean(S; dims = 1)); color = PALETTE.red, linestyle = :dash, linewidth = 1.2,
       label = "sample mean of 2,000 paths")
xlims!(ax, 0, T); ylims!(ax, 1e-3, 1e2)
axislegend(ax; position = :lb, nbanks = 2, labelsize = 12)
save_fig(fig, @__DIR__, "gbm")
