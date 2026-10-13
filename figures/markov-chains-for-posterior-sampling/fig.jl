# Gibbs sampling and random-walk Metropolis for a normal model with semi-conjugate prior.
# Packages: CairoMakie, Distributions, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()

# Data and prior: y_i ~ N(mu, 1/tau); mu ~ N(0, 10^2); tau ~ Gamma(shape 2, rate 1).
rng = MersenneTwister(2024)
n = 12
y = 1.5 .+ 0.8 .* randn(rng, n)
a0, b0, kappa0 = 2.0, 1.0, 1 / 100

log_post(mu, eta) = (n / 2 + a0) * eta - exp(eta) * (b0 + sum((y .- mu) .^ 2) / 2) - kappa0 * mu^2 / 2

function gibbs(T, mu, tau, rng)
    out = zeros(2T + 1, 2); out[1, :] = [mu, log(tau)]; k = 2
    for _ in 1:T
        prec = tau * n + kappa0
        mu = rand(rng, Normal(tau * sum(y) / prec, 1 / sqrt(prec))); out[k, :] = [mu, log(tau)]; k += 1
        tau = rand(rng, Gamma(a0 + n / 2, 1 / (b0 + sum((y .- mu) .^ 2) / 2))); out[k, :] = [mu, log(tau)]; k += 1
    end
    return out
end

function rwm(T, x, h, rng)
    out = zeros(T + 1, 2); out[1, :] = x; lp = log_post(x...); acc = 0
    for t in 1:T
        prop = x .+ h .* randn(rng, 2)
        lq = log_post(prop...)
        if log(rand(rng)) < lq - lp
            x, lp = prop, lq; acc += 1
        end
        out[t + 1, :] = x
    end
    return out, acc / T
end

# ---- Figure 1: first moves of each sampler over posterior contours
ms = range(-1.2, 3.6; length = 300); es = range(-2.4, 1.4; length = 300)
LP = [log_post(m, e) for m in ms, e in es]
P = exp.(LP .- maximum(LP))
start = [-0.3, -2.0]
g = gibbs(25, start[1], exp(start[2]), MersenneTwister(1))
r, _ = rwm(50, start, 0.35, MersenneTwister(2))

fig = Figure(size = (760, 350))
for (j, (path, col, title)) in enumerate([(g, PALETTE.blue, "Gibbs sampler, 25 sweeps"),
                                          (r, PALETTE.red, "Random-walk Metropolis, 50 steps")])
    ax = Axis(fig[1, j]; title = title, xlabel = L"\mu", ylabel = j == 1 ? L"\eta = \log \tau" : "")
    contour!(ax, ms, es, P; levels = [0.02, 0.1, 0.3, 0.6, 0.9], color = PALETTE.grey, linewidth = 1)
    lines!(ax, path[:, 1], path[:, 2]; color = (col, 0.85), linewidth = 1.2)
    scatter!(ax, path[:, 1], path[:, 2]; color = col, markersize = 5)
    scatter!(ax, [start[1]], [start[2]]; marker = :rect, color = PALETTE.ink, markersize = 10)
    xlims!(ax, -1.2, 3.6); ylims!(ax, -2.4, 1.4)
    j == 2 && hideydecorations!(ax; grid = false)
end
save_fig(fig, @__DIR__, "sampler-paths")

# ---- Figure 2: traces and autocorrelations of mu for three proposal scales
function acf(x, K)
    z = x .- mean(x); v = mean(z .^ 2)
    return [mean(z[1:end-k] .* z[1+k:end]) / v for k in 0:K]
end

rng = MersenneTwister(7)
hs = [0.05, 0.5, 5.0]
cols = [PALETTE.gold, PALETTE.blue, PALETTE.red]
runs = [rwm(4000, [1.5, 0.0], h, rng) for h in hs]

fig = Figure(size = (780, 440))
ax_acf = Axis(fig[1:3, 2]; title = "Autocorrelation", xlabel = L"\text{lag } k",
              ylabel = L"\text{autocorrelation of } \mu_t")
trace_axes = Axis[]
for (i, (h, col, (path, acc))) in enumerate(zip(hs, cols, runs))
    ax = Axis(fig[i, 1]; title = @sprintf("h = %s, acceptance rate %.2f", h, acc), titlesize = 13,
              ylabel = L"\mu_t", yticks = [0.5, 1.5, 2.5], xlabel = i == 3 ? L"\text{iteration } t" : "")
    lines!(ax, 0:1000, path[1:1001, 1]; color = col, linewidth = 1)
    xlims!(ax, 0, 1000); ylims!(ax, 0.2, 2.8)
    i < 3 && hidexdecorations!(ax; grid = false)
    push!(trace_axes, ax)
    lines!(ax_acf, 0:60, acf(path[1001:end, 1], 60); color = col, label = "h = $h")
end
linkxaxes!(trace_axes...)
hlines!(ax_acf, [0.0]; color = PALETTE.ink, linewidth = 0.8)
axislegend(ax_acf; position = :rt)
colsize!(fig.layout, 1, Relative(0.66))
save_fig(fig, @__DIR__, "step-size")
