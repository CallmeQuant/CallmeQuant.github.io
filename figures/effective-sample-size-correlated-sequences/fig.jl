# Effective sample size for AR(1) sequences: variance inflation and estimation of tau.
# Packages: CairoMakie, Random, Statistics, FFTW
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, FFTW

set_blog_theme!()
rng = MersenneTwister(1992)

function ar1(phi, n, m = 1)                # m stationary paths (rows), unit marginal variance
    x = zeros(m, n)
    x[:, 1] = randn(rng, m)
    s = sqrt(1 - phi^2)
    for t in 2:n
        x[:, t] = phi .* x[:, t-1] .+ s .* randn(rng, m)
    end
    return x
end
inflation(phi, n) = (1 + phi) / (1 - phi) - 2phi * (1 - phi^n) / (n * (1 - phi)^2)

phis = [-0.5, 0.5, 0.9]
cols = [PALETTE.green, PALETTE.blue, PALETTE.red]
fig = Figure(size = (920, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"t", title = "Stationary AR(1) paths, unit variance",
           yticksvisible = false, yticklabelsvisible = false, xgridvisible = false, ygridvisible = false)
for (j, (phi, col)) in enumerate(zip(phis, cols))
    lines!(ax1, 1:200, ar1(phi, 200)[1, :] .+ 5 * (3 - j); color = col, linewidth = 1)
    text!(ax1, 203, 5 * (3 - j); text = L"\phi = %$phi", color = col, align = (:left, :center), fontsize = 13)
end
xlims!(ax1, 0, 235)

ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"n",
           ylabel = L"n\,\mathrm{Var}(\bar X_n)/\gamma_0",
           title = L"\text{Variance inflation approaches } \tau = (1+\phi)/(1-\phi)")
ns = unique(round.(Int, 10 .^ range(0, 3.3; length = 60)))
hlines!(ax2, [1.0]; color = PALETTE.ink, linewidth = 0.8)
for (phi, col) in zip(phis, cols)
    hlines!(ax2, [(1 + phi) / (1 - phi)]; xmin = 0.4, color = col, linestyle = :dot, linewidth = 0.8)
    lines!(ax2, ns, inflation.(phi, ns); color = col, label = L"\phi = %$phi")
    pts = [n * var(vec(mean(ar1(phi, n, 4000); dims = 2))) for n in [5, 20, 100, 500]]
    scatter!(ax2, [5, 20, 100, 500], pts; color = :white, strokecolor = col, strokewidth = 1.2, markersize = 8)
end
xlims!(ax2, 1, 2000); ylims!(ax2, 0.2, 40)
axislegend(ax2; position = :lt, framevisible = false)
save_fig(fig, @__DIR__, "variance-inflation")

function acov(x)
    n = length(x); y = x .- mean(x)
    f = rfft(vcat(y, zeros(n)))
    return irfft(abs2.(f), 2n)[1:n] ./ n
end
function tau_geyer(x)
    g = acov(x); m = length(g) ÷ 2
    G = g[1:2:2m] .+ g[2:2:2m]
    k = findfirst(<=(0), G)
    M = k === nothing ? m : k - 1
    return (-g[1] + 2sum(G[1:M])) / g[1]
end
function tau_batch(x)
    n = length(x); m = floor(Int, sqrt(n)); b = n ÷ m
    bm = vec(mean(reshape(x[1:b*m], m, b); dims = 1))
    return m * var(bm) / mean((x .- mean(x)) .^ 2)
end

phi, n = 0.9, 5000
tau_true = (1 + phi) / (1 - phi)
fig = Figure(size = (920, 330))
bx1 = Axis(fig[1, 1]; xscale = log10, xlabel = L"\text{truncation lag } K",
           title = L"\text{One chain, } \phi = 0.9,\ n = 5000")
x = ar1(phi, n)[1, :]
g = acov(x)
partial = (g[1] .+ 2 .* cumsum(g[2:end])) ./ g[1]
hlines!(bx1, [0.0]; color = PALETTE.ink, linewidth = 0.8)
lines!(bx1, 1:n-1, partial; color = PALETTE.blue, linewidth = 1.2, label = L"1 + 2\sum_{k \le K}\hat\rho_k")
hlines!(bx1, [tau_true]; color = PALETTE.red, linestyle = :dash, label = L"\text{true } \tau = 19")
xlims!(bx1, 1, n)
axislegend(bx1; position = :lb)

bx2 = Axis(fig[1, 2]; xlabel = L"\text{estimated } n_{\mathrm{eff}}", ylabel = "chains", title = "Estimates from 2,000 chains")
X = ar1(phi, n, 2000)
ess_b = [n / tau_batch(X[i, :]) for i in 1:2000]
ess_g = [n / tau_geyer(X[i, :]) for i in 1:2000]
bins = range(100, 600; length = 51)
hist!(bx2, ess_b; bins = bins, color = (PALETTE.gold, 0.55), label = "batch means")
hist!(bx2, ess_g; bins = bins, color = (PALETTE.blue, 0.55), label = "initial positive sequence")
vlines!(bx2, [n / tau_true]; color = PALETTE.red, linestyle = :dash, linewidth = 1.4, label = L"n/\tau = 263")
axislegend(bx2; position = :rt)
save_fig(fig, @__DIR__, "estimating-tau")
