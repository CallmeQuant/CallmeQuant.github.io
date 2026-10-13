# Quasi-likelihood with a misspecified variance function: gamma-type data with Var(Y) = phi mu^2 and
# log mu = b0 + b1 x, fitted by quasi-score equations with working variance V(mu) = mu^q.
# Packages: CairoMakie, Distributions, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, LinearAlgebra, Statistics

set_blog_theme!()
rng = MersenneTwister(1974)
const BETA = [0.5, 1.0]; const PHI = 0.5; const n = 200
x = collect(range(0, 3; length = n)); X = hcat(ones(n), x); mu0 = exp.(X * BETA)

function asym_cov(q; mu = mu0)
    D = X .* mu; A = D ./ mu .^ q
    bread = inv(A' * D)
    return bread * ((A .* (PHI .* mu .^ 2))' * A) * bread'
end

function quasi_fit(y, q)
    b = [log(mean(y)), 0.0]
    for _ in 1:50
        mu = exp.(X * b); D = X .* mu; A = D ./ mu .^ q
        step = (A' * D) \ (A' * (y .- mu)); b .+= step
        maximum(abs.(step)) < 1e-10 && break
    end
    mu = exp.(X * b); D = X .* mu; A = D ./ mu .^ q
    bread = inv(A' * D)
    phi_hat = sum((y .- mu) .^ 2 ./ mu .^ q) / (n - 2)
    se_m = sqrt(phi_hat * bread[2, 2])
    se_s = sqrt((bread * ((A .* (y .- mu) .^ 2)' * A) * bread')[2, 2])
    return b, se_m, se_s
end

qs_fine = range(0, 3; length = 121)
v_opt = asym_cov(2.0)[2, 2]
qs = 0:0.5:3; R = 2000; z = quantile(Normal(), 0.975)
Y = [rand(rng, Gamma(1 / PHI, mu0[i] * PHI)) for r in 1:R, i in 1:n]   # mean mu0, variance PHI mu0^2
vs, cm, cs = Float64[], Float64[], Float64[]
for q in qs
    fits = [quasi_fit(Y[r, :], q) for r in 1:R]
    b1 = [f[1][2] for f in fits]
    push!(vs, var(b1))
    push!(cm, mean(abs.(b1 .- BETA[2]) .<= z .* [f[2] for f in fits]))
    push!(cs, mean(abs.(b1 .- BETA[2]) .<= z .* [f[3] for f in fits]))
end
sim_eff = vs[findfirst(==(2.0), collect(qs))] ./ vs

fig = Figure(size = (780, 340))
ax1 = Axis(fig[1, 1]; title = "Efficiency", xlabel = L"\text{working variance power } q \text{ in } V(\mu) = \mu^q",
           ylabel = L"\text{efficiency of } \hat\beta_1 \text{ relative to } q = 2")
vlines!(ax1, [2.0]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
lines!(ax1, qs_fine, [v_opt / asym_cov(q)[2, 2] for q in qs_fine]; color = PALETTE.blue, label = "asymptotic")
scatter!(ax1, collect(qs), sim_eff; color = :white, strokecolor = PALETTE.blue, strokewidth = 1.5,
         markersize = 9, label = L"\text{simulation, } n = 200")
ylims!(ax1, 0, 1.05)
axislegend(ax1; position = :rb, labelsize = 12)

ax2 = Axis(fig[1, 2]; title = L"\text{Confidence intervals for } \beta_1", xlabel = L"\text{working variance power } q",
           ylabel = "coverage of nominal 95% interval")
hlines!(ax2, [0.95]; color = PALETTE.ink, linewidth = 0.8)
vlines!(ax2, [2.0]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
scatterlines!(ax2, collect(qs), cm; color = PALETTE.red, marker = :circle, markercolor = :white,
              strokecolor = PALETTE.red, strokewidth = 1.4, markersize = 9,
              label = L"\text{model-based: } \hat\phi\,(D^\top V^{-1} D)^{-1}")
scatterlines!(ax2, collect(qs), cs; color = PALETTE.blue, marker = :rect, markercolor = :white,
              strokecolor = PALETTE.blue, strokewidth = 1.4, markersize = 9, label = "sandwich")
ylims!(ax2, 0.7, 1.0)
axislegend(ax2; position = :cb, labelsize = 12)
save_fig(fig, @__DIR__, "working-variance")
