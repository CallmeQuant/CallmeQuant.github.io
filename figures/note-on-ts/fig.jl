# Sample paths and sample autocorrelation functions for Gaussian white noise, an AR(1) process
# with phi = 0.7 and an MA(1) process with theta = 0.8.
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics

set_blog_theme!()
rng = MersenneTwister(3)
n, H, burn = 200, 20, 500
phi, theta = 0.7, 0.8
z = randn(rng, n + burn)
ar = zeros(n + burn)
for t in 2:n+burn
    ar[t] = phi * ar[t-1] + z[t]
end
idx = burn+1:burn+n
xs = [z[idx], ar[idx], z[idx] .+ theta .* z[idx .- 1]]
rhos = [zeros(H), phi .^ (1:H), vcat(theta / (1 + theta^2), zeros(H - 1))]
titles = [L"\text{white noise}", L"\text{AR(1)},\ \phi = 0.7", L"\text{MA(1)},\ \theta = 0.8"]

function sample_acf(x, H)
    xc = x .- mean(x); m = length(x); g0 = sum(abs2, xc) / m
    [sum(xc[h+1:end] .* xc[1:end-h]) / m / g0 for h in 1:H]
end

fig = Figure(size = (820, 470))
band = 1.96 / sqrt(n)
for j in 1:3
    ax = Axis(fig[1, j]; title = titles[j], xlabel = L"t", ylabel = j == 1 ? L"X_t" : "")
    lines!(ax, 1:n, xs[j]; color = PALETTE.blue, linewidth = 0.9)
    ax2 = Axis(fig[2, j]; xlabel = L"\text{lag } h", ylabel = j == 1 ? "autocorrelation" : "",
               xticks = [1, 5, 10, 15, 20])
    hspan!(ax2, -band, band; color = PALETTE.light)
    hlines!(ax2, [0.0]; color = PALETTE.ink, linewidth = 0.8)
    r = sample_acf(xs[j], H)
    rangebars!(ax2, 1:H, zeros(H), r; color = PALETTE.blue, linewidth = 2.2, whiskerwidth = 0,
               label = L"\text{sample } \hat\rho(h)")
    scatter!(ax2, 1:H, rhos[j]; color = :white, strokecolor = PALETTE.red, strokewidth = 1.5,
             markersize = 8, label = L"\text{true } \rho(h)")
    ylims!(ax2, -0.35, 0.85)
    j == 3 && axislegend(ax2; position = :rt)
end
save_fig(fig, @__DIR__, "sample-acf")
