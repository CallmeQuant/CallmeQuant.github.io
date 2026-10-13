# Consistency of maximum likelihood: Cauchy location criterion and MLE paths (figure 1);
# unbounded likelihood of a normal mixture with a free scale (figure 2).
# Packages: CairoMakie, Distributions, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1949)

avg_llr(x, t) = -mean(log1p.((x .- t) .^ 2) .- log1p.(x .^ 2))

function golden_max(f, a, b; iters = 60)
    g = (sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in 1:iters
        if f(c) > f(d)
            b, d = d, c; c = b - g * (b - a)
        else
            a, c = c, d; d = a + g * (b - a)
        end
    end
    return (a + b) / 2
end

function cauchy_mle(x)
    grid = range(-15, 15; length = 3001)
    g = grid[argmax([avg_llr(x, t) for t in grid])]
    return golden_max(t -> avg_llr(x, t), g - 0.02, g + 0.02)
end

# ---- Figure 1 ----
x = rand(rng, Cauchy(0, 1), 5000)
th = range(-6, 6; length = 1201)
fig = Figure(size = (780, 340))
ax1 = Axis(fig[1, 1]; title = L"\text{Criterion and its limit } (\theta_0 = 0)", xlabel = L"\theta",
           ylabel = "average log-likelihood ratio")
for (col, n) in zip([PALETTE.gold, PALETTE.green, PALETTE.blue], [10, 100, 1000])
    lines!(ax1, th, [avg_llr(x[1:n], t) for t in th]; color = col, linewidth = 1.6,
           label = L"M_n(\theta),\ n = %$n")
end
lines!(ax1, th, -log1p.(th .^ 2 ./ 4); color = PALETTE.ink, linestyle = :dash,
       label = L"M(\theta) = -\log(1 + \theta^2/4)")
xlims!(ax1, -6, 6); ylims!(ax1, -2.6, 1.3)
axislegend(ax1; position = :lt, labelsize = 12)

ax2 = Axis(fig[1, 2]; title = "Five paths of the global maximizer", xlabel = "sample size n (log scale)",
           ylabel = L"\hat\theta_n", xscale = log10)
ns = unique(round.(Int, 10 .^ range(0.7, log10(5000); length = 70)))
for j in 1:5
    xs = j == 1 ? x : rand(rng, Cauchy(0, 1), 5000)
    lines!(ax2, ns, [cauchy_mle(xs[1:n]) for n in ns]; color = CYCLE[j], linewidth = 1.3)
end
hlines!(ax2, [0.0]; color = PALETTE.ink, linewidth = 0.8)
ylims!(ax2, -3, 3)
colsize!(fig.layout, 1, Relative(0.53))
save_fig(fig, @__DIR__, "cauchy-criterion")

# ---- Figure 2 ----
n = 200
z = rand(rng, n) .< 0.5
y = [zi ? 2.0 + randn(rng) : randn(rng) for zi in z]

function loglik(mu, logsig)
    c = -0.5 * log(2pi) + log(0.5)
    a = c .- y .^ 2 ./ 2
    q = exp.(min.(2 .* log.(abs.(y .- mu)) .- 2logsig, 700.0))
    b = c .- logsig .- 0.5 .* q
    m = max.(a, b)
    return sum(m .+ log.(exp.(a .- m) .+ exp.(b .- m)))
end

fig2 = Figure(size = (780, 340))
Label(fig2[0, 1:2], L"\text{Model } \frac{1}{2}N(0,1) + \frac{1}{2}N(\mu,\sigma^2),\ \text{data with } \mu = 2,\ \sigma = 1,\ n = 200";
      halign = :left, fontsize = 15)
lmax = maximum(loglik(2.0, t) for t in range(-2.0, 1.5; length = 3501))   # local maximum of the mu = 2 curve
axs = Axis[]
for (k, (lo, npts, ttl)) in enumerate([(-8.0, 600, "Near the true scale"), (-500.0, 2000, L"\text{Down to } \sigma = e^{-500}")])
    ax = Axis(fig2[1, k]; title = ttl, xlabel = L"\log \sigma", ylabel = k == 1 ? L"\text{log-likelihood } \ell_n(\mu,\sigma)" : "")
    ls = range(lo, 1.5; length = npts)
    hlines!(ax, [lmax]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
    lines!(ax, ls, [loglik(2.0, t) for t in ls]; color = PALETTE.blue, label = L"\mu = 2\ \text{(true value)}")
    lines!(ax, ls, [loglik(y[1], t) for t in ls]; color = PALETTE.red, label = L"\mu = Y_1")
    k == 1 && axislegend(ax; position = :lt, labelsize = 12)
    push!(axs, ax)
end
linkyaxes!(axs...)
hideydecorations!(axs[2]; grid = false, ticks = false)
save_fig(fig2, @__DIR__, "mixture-degeneracy")
