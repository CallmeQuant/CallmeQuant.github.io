# Random Fourier features: Monte Carlo approximations of the Gaussian and Laplace kernels, and the error as
# a function of the number of features.
# Packages: CairoMakie, Random, Statistics, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Distributions

set_blog_theme!()
rng = MersenneTwister(2007)
const ELL = 1.0
delta = collect(range(-6, 6; length = 601))
kg(d) = exp(-0.5 * d^2 / ELL^2)
kl(d) = exp(-abs(d) / ELL)
rff(d::AbstractVector, om, b) = vec(mean(2 .* cos.(d .* om' .+ b') .* cos.(b'); dims = 2))

fig = Figure(size = (780, 310))
specs = [(kg, D -> rand(rng, Normal(0, 1 / ELL), D), L"\text{Gaussian kernel, } \omega \sim \mathcal{N}(0, \ell^{-2})"),
         (kl, D -> rand(rng, Cauchy(0, 1 / ELL), D), L"\text{Laplace kernel, } \omega \sim \mathrm{Cauchy}(0, \ell^{-1})")]
axs = Axis[]
for (j, (kf, sampler, ttl)) in enumerate(specs)
    ax = Axis(fig[1, j]; title = ttl, xlabel = L"\delta = x - x'", ylabel = j == 1 ? L"\hat k_D(\delta)" : "")
    lines!(ax, delta, kf.(delta); color = PALETTE.ink, linewidth = 2.6, label = "exact")
    for (c, D) in zip([PALETTE.gold, PALETTE.blue], [20, 500])
        om = sampler(D); b = 2π .* rand(rng, D)
        lines!(ax, delta, rff(delta, om, b); color = c, linewidth = 1.4, label = L"D = %$D")
    end
    xlims!(ax, -6, 6); ylims!(ax, -0.65, 1.3)
    push!(axs, ax)
end
hideydecorations!(axs[2]; grid = false)
axislegend(axs[1]; position = :lt, labelsize = 12)
save_fig(fig, @__DIR__, "kernel-approximations")

Ds = unique(div.(round.(Int, exp.(range(log(4), log(4096); length = 12))), 2) .* 2)
reps = 400; d0 = 1.0; k0 = kg(d0); k2 = kg(2d0)
grid = collect(range(-4, 4; length = 161))
sd_ph, sd_pr, sup_ph, sup_pr = Float64[], Float64[], Float64[], Float64[]
for D in Ds
    e1, e2, s1, s2 = Float64[], Float64[], Float64[], Float64[]
    for _ in 1:reps
        om = rand(rng, Normal(0, 1 / ELL), D); b = 2π .* rand(rng, D)
        push!(e1, rff([d0], om, b)[1] - k0)
        push!(s1, maximum(abs.(rff(grid, om, b) .- kg.(grid))))
        om2 = rand(rng, Normal(0, 1 / ELL), D ÷ 2)
        push!(e2, mean(cos.(om2 .* d0)) - k0)
        push!(s2, maximum(abs.(vec(mean(cos.(grid .* om2'); dims = 2)) .- kg.(grid))))
    end
    push!(sd_ph, std(e1)); push!(sd_pr, std(e2)); push!(sup_ph, mean(s1)); push!(sup_pr, mean(s2))
end
fig = Figure(size = (780, 320))
ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = L"\text{number of features } D", ylabel = "error",
           title = L"\text{Standard deviation of } \hat k_D(\delta),\ \delta = 1")
scatter!(ax1, Ds, sd_ph; color = PALETTE.red, label = "random phase (simulated)")
lines!(ax1, Ds, sqrt.((1 + 0.5k2 - k0^2) ./ Ds); color = PALETTE.red, label = "random phase (formula)")
scatter!(ax1, Ds, sd_pr; color = PALETTE.blue, marker = :rect, label = "(cos, sin) pairs (simulated)")
lines!(ax1, Ds, sqrt.((1 + k2 - 2k0^2) ./ Ds); color = PALETTE.blue, label = "(cos, sin) pairs (formula)")
axislegend(ax1; position = :lb, labelsize = 11)
ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{number of features } D",
           title = L"\text{Mean of } \sup |\hat k_D - k| \text{ over } |\delta| \leq 4")
scatterlines!(ax2, Ds, sup_ph; color = PALETTE.red, label = "random phase")
scatterlines!(ax2, Ds, sup_pr; color = PALETTE.blue, marker = :rect, label = "(cos, sin) pairs")
lines!(ax2, Ds, 2.5 ./ sqrt.(Ds); color = PALETTE.grey, linestyle = :dash, label = "slope −1/2")
axislegend(ax2; position = :lb, labelsize = 11)
linkyaxes!(ax1, ax2)
save_fig(fig, @__DIR__, "error-vs-features")
