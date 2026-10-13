# Maximum likelihood under misspecification.
# Figure 1: normal linear model fitted to a nonlinear regression; the projection depends on the design.
# Figure 2: exponential model fitted to log-normal data; coverage of model-based and sandwich intervals.
# Packages: CairoMakie, Distributions, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1982)

# ---- Figure 1 ----
m(z) = 2sin(z)
n = 300
zz = range(-3.2, 3.2; length = 400)
fig = Figure(size = (780, 360))
axs = Axis[]
for (k, (lab, mu, sd)) in enumerate([("Design A: Z ~ N(0, 1)", 0.0, 1.0), ("Design B: Z ~ N(1.5, 0.5²)", 1.5, 0.5)])
    z = rand(rng, Normal(mu, sd), n)
    y = m.(z) .+ 0.5 .* randn(rng, n)
    X = hcat(ones(n), z)
    b = X \ y
    slope = 2cos(mu) * exp(-sd^2 / 2)
    inter = 2sin(mu) * exp(-sd^2 / 2) - slope * mu
    ax = Axis(fig[1, k]; title = @sprintf("%s, β₁* = %.2f", lab, slope), titlesize = 13,
              xlabel = L"z", ylabel = k == 1 ? L"y" : "")
    dens = pdf.(Normal(mu, sd), zz)
    band!(ax, zz, fill(-4.6, length(zz)), -4.6 .+ 1.3 .* dens ./ maximum(dens); color = (PALETTE.gold, 0.35))
    scatter!(ax, z, y; color = (PALETTE.grey, 0.55), markersize = 4)
    lines!(ax, zz, m.(zz); color = PALETTE.ink, linewidth = 1.8, label = L"\mathrm{E}(Y \mid Z = z) = 2\sin z")
    lines!(ax, zz, inter .+ slope .* zz; color = PALETTE.blue, linewidth = 2, label = L"\text{projection } \beta_0^* + \beta_1^* z")
    lines!(ax, zz, b[1] .+ b[2] .* zz; color = PALETTE.red, linewidth = 1.5, linestyle = :dash,
           label = L"\text{fit } \hat\beta_0 + \hat\beta_1 z,\ n = 300")
    xlims!(ax, -3.2, 3.2); ylims!(ax, -4.6, 4.6)
    k == 1 && axislegend(ax; position = :lt, labelsize = 11)
    push!(axs, ax)
end
linkyaxes!(axs...)
save_fig(fig, @__DIR__, "design-dependence")

# ---- Figure 2 ----
ss = 0.2:0.1:1.6
z975 = quantile(Normal(), 0.975)
sfine = range(0.15, 1.65; length = 300)
fig2 = Figure(size = (660, 380))
ax = Axis(fig2[1, 1]; xlabel = L"\text{log-normal shape } s", ylabel = "coverage of nominal 95% interval")
vlines!(ax, [sqrt(log(2))]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
text!(ax, sqrt(log(2)) + 0.02, 0.76; text = L"\mathrm{CV} = 1", color = PALETTE.grey, fontsize = 13)
lines!(ax, sfine, 2 .* cdf.(Normal(), z975 ./ sqrt.(expm1.(sfine .^ 2))) .- 1; color = PALETTE.red,
       label = L"\text{model-based, } n \to \infty")
hlines!(ax, [0.95]; color = PALETTE.blue, linewidth = 2.2, label = L"\text{sandwich, } n \to \infty")
R = 20000
for (nn, mk) in [(100, :circle), (1000, :rect)]
    cm, cs = Float64[], Float64[]
    for s in ss
        lam_star = exp(-s^2 / 2)
        hits_m = 0; hits_s = 0
        for _ in 1:R
            x = rand(rng, LogNormal(0, s), nn)
            xbar = mean(x); lam = 1 / xbar
            hits_m += abs(lam - lam_star) <= z975 * lam / sqrt(nn)
            hits_s += abs(lam - lam_star) <= z975 * sqrt(var(x) / nn) / xbar^2
        end
        push!(cm, hits_m / R); push!(cs, hits_s / R)
    end
    scatter!(ax, collect(ss), cm; marker = mk, color = :white, strokecolor = PALETTE.red, strokewidth = 1.3,
             markersize = 8, label = "model-based, n = $nn")
    scatter!(ax, collect(ss), cs; marker = mk, color = :white, strokecolor = PALETTE.blue, strokewidth = 1.3,
             markersize = 8, label = "sandwich, n = $nn")
end
xlims!(ax, 0.15, 1.65); ylims!(ax, 0.38, 1.02)
axislegend(ax; position = :lb, labelsize = 11)
save_fig(fig2, @__DIR__, "coverage")
