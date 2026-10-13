# EM for the symmetric mixture 0.5 N(theta, 1) + 0.5 N(-theta, 1): lower bounds touching the
# log-likelihood, and linear convergence at the missing-information rate.
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1977)
n = 200
simulate(ts) = rand(rng, [-1.0, 1.0], n) .* ts .+ randn(rng, n)
lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
loglik(t, x) = sum(lse.(-0.5 .* (x .- t) .^ 2, -0.5 .* (x .+ t) .^ 2) .- log(2) .- 0.5 * log(2π))
function lower_bound(t, x, tt)
    r = 1 ./ (1 .+ exp.(-2 * tt .* x))
    ent = -(r .* log.(r) .+ (1 .- r) .* log1p.(-r))
    return sum(r .* (-0.5 .* (x .- t) .^ 2) .+ (1 .- r) .* (-0.5 .* (x .+ t) .^ 2) .- log(2) .- 0.5 * log(2π) .+ ent)
end
em_map(t, x) = mean(x .* tanh.(t .* x))

x = simulate(1.0)
it = [0.15]
for k in 1:4
    push!(it, em_map(it[end], x))
end
grid = collect(range(-0.2, 1.8; length = 400))
ll = [loglik(t, x) for t in grid]

fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"\theta", ylabel = "log-likelihood and lower bounds",
           title = L"\text{Bounds touching } \ell \text{ at } \theta_t")
lines!(ax1, grid, ll; color = PALETTE.ink, linewidth = 2.2, label = L"\ell(\theta)")
for t in 1:4
    lines!(ax1, grid, [lower_bound(g, x, it[t]) for g in grid]; color = CYCLE[t], linestyle = :dash,
           linewidth = 1.6, label = L"F(q_{%$t}, \theta)")
    scatter!(ax1, [it[t]], [loglik(it[t], x)]; color = CYCLE[t], markersize = 8)
    scatter!(ax1, [it[t+1]], [lower_bound(it[t+1], x, it[t])]; color = CYCLE[t], marker = :utriangle, markersize = 9)
end
xlims!(ax1, -0.2, 1.8); ylims!(ax1, maximum(ll) - 70, maximum(ll) + 8)
axislegend(ax1; position = :lt, labelsize = 12)

ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{iteration } t", ylabel = L"|\theta_t - \hat\theta|",
           title = "Linear convergence")
for (k, ts) in enumerate([0.6, 1.0, 2.0])
    xs = simulate(ts)
    th = ts > 0.7 ? 0.5 * ts : 1.2
    h = [th]
    for _ in 1:400
        push!(h, em_map(h[end], xs))
    end
    hat = h[end]
    rate = mean(xs .^ 2 ./ cosh.(hat .* xs) .^ 2)
    err = abs.(h[1:60] .- hat)
    T = collect(0:59)
    keep = err .> 1e-13
    scatter!(ax2, T[keep], err[keep]; color = CYCLE[k], markersize = 6,
                  label = L"\theta^\ast = %$(ts):\ \hat\rho = %$(round(rate; digits = 2))")
    lines!(ax2, T, err[4] .* rate .^ (T .- 3); color = CYCLE[k], linestyle = :dot, linewidth = 1.2)
end
xlims!(ax2, 0, 59); ylims!(ax2, 1e-12, 3)
axislegend(ax2; position = :rt, labelsize = 12)
save_fig(fig, @__DIR__, "em-bounds-and-rates")
