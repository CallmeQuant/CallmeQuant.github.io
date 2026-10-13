# Wiener process, Brownian motion with drift, an Ornstein-Uhlenbeck (Ito) process;
# variation of Brownian paths on finer grids.
# Packages: CairoMakie, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random

set_blog_theme!()
rng = MersenneTwister(1923)
T, N, paths = 4.0, 2000, 5
dt = T / N
t = collect(range(0, T; length = N + 1))
dW = sqrt(dt) .* randn(rng, paths, N)
W = hcat(zeros(paths), cumsum(dW; dims = 2))
a, b = 0.75, 0.6
kappa, sigma, x0 = 2.0, 0.8, 1.5
X = a .* t' .+ b .* W
Y = zeros(paths, N + 1); Y[:, 1] .= x0
for k in 1:N                                   # Euler-Maruyama
    Y[:, k + 1] = Y[:, k] .- kappa .* Y[:, k] .* dt .+ sigma .* dW[:, k]
end

m = x0 .* exp.(-kappa .* t)
s = sigma .* sqrt.((1 .- exp.(-2kappa .* t)) ./ (2kappa))
specs = [(W, zeros(N + 1), 2 .* sqrt.(t), L"W_t\text{: Wiener process}"),
         (X, a .* t, 2b .* sqrt.(t), L"X_t = 0.75\,t + 0.6\,W_t"),
         (Y, m, 2 .* s, L"dY_t = -2\,Y_t\,dt + 0.8\,dW_t")]
fig = Figure(size = (980, 330))
for (j, (M, mid, half, title)) in enumerate(specs)
    ax = Axis(fig[1, j]; title = title, xlabel = L"t")
    band!(ax, t, mid .- half, mid .+ half; color = PALETTE.light)
    lines!(ax, t, mid; color = PALETTE.ink, linestyle = :dash, linewidth = 1.2)
    for i in 1:paths
        lines!(ax, t, M[i, :]; color = CYCLE[i], linewidth = 1)
    end
    xlims!(ax, 0, T)
end
save_fig(fig, @__DIR__, "three-processes")

K = 2^20
levels = 2:20
fig = Figure(size = (820, 310))
ax1 = Axis(fig[1, 1]; xscale = log2, xlabel = L"\text{number of intervals } n",
           title = "Sum of squared increments on [0, 1]")
ax2 = Axis(fig[1, 2]; xscale = log2, yscale = log10, xlabel = L"\text{number of intervals } n",
           title = "Sum of absolute increments on [0, 1]")
for i in 1:3
    path = vcat(0.0, cumsum(sqrt(1 / K) .* randn(rng, K)))
    qv = Float64[]; tv = Float64[]
    for l in levels
        d = diff(path[1:2^(20 - l):end])
        push!(qv, sum(d .^ 2)); push!(tv, sum(abs.(d)))
    end
    scatterlines!(ax1, 2.0 .^ levels, qv; color = CYCLE[i], markersize = 6, linewidth = 1.2)
    scatterlines!(ax2, 2.0 .^ levels, tv; color = CYCLE[i], markersize = 6, linewidth = 1.2)
end
hlines!(ax1, [1.0]; color = PALETTE.ink, linestyle = :dash, linewidth = 1)
n = 2.0 .^ levels
lines!(ax2, n, sqrt.(2 .* n ./ pi); color = PALETTE.ink, linestyle = :dash, linewidth = 1, label = L"\sqrt{2n/\pi}")
ylims!(ax1, 0, 2.6)
axislegend(ax2; position = :lt)
save_fig(fig, @__DIR__, "variation")
