# Normalizing flows: change of variables in one dimension (three examples) and two affine
# coupling layers applied to a standard Gaussian in the plane.
# Packages: CairoMakie, Distributions, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random

set_blog_theme!()
rng = MersenneTwister(1505)
n = 100_000
phi(u) = pdf(Normal(), u)

# ------------------------------------------------------------------- one dimension
fig = Figure(size = (880, 300))
histkw = (bins = 80, normalization = :pdf, color = PALETTE.light, strokecolor = :white, strokewidth = 0.3)

ax = Axis(fig[1, 1]; xlabel = L"x", ylabel = "density", title = L"\text{(a) } U \sim \mathrm{Unif}(0,1),\ X = U^2")
hist!(ax, rand(rng, n) .^ 2; histkw...)
xs = range(1e-3, 1, length = 400)
lines!(ax, xs, 1 ./ (2 .* sqrt.(xs)); color = PALETTE.blue, label = L"1/(2\sqrt{x})")
xlims!(ax, 0, 1); ylims!(ax, 0, 5)
axislegend(ax; position = :rt)

Tm(u) = u + 2 * tanh(2u)
dT(u) = 1 + 4 / cosh(2u)^2
ax = Axis(fig[1, 2]; xlabel = L"x", title = L"\text{(b) } X = U + 2\tanh(2U),\ U \sim \mathcal{N}(0,1)")
xb = Tm.(randn(rng, n))
hist!(ax, xb[abs.(xb) .<= 6]; histkw...)
us = range(-6, 6, length = 800)
lines!(ax, Tm.(us), phi.(us) ./ dT.(us); color = PALETTE.blue, label = L"\varphi(u)/T'(u)")
lines!(ax, us, phi.(us); color = PALETTE.grey, linestyle = :dash, linewidth = 1.5, label = L"\text{base } \varphi")
xlims!(ax, -6, 6); ylims!(ax, 0, 0.55)
axislegend(ax; position = :rt)

ax = Axis(fig[1, 3]; xlabel = L"x", title = L"\text{(c) } X = U^2,\ U \sim \mathcal{N}(0,1)")
xc = randn(rng, n) .^ 2
keep = xc[xc .<= 5]      # density relative to all n draws, on 80 bins of width 5/80
hist!(ax, keep; bins = range(0, 5, length = 81), normalization = :none,
      weights = fill(1 / (n * 5 / 80), length(keep)),
      color = PALETTE.light, strokecolor = :white, strokewidth = 0.3)
xs = range(0.02, 5, length = 400)
lines!(ax, xs, pdf.(Chisq(1), xs); color = PALETTE.blue, label = "sum over both preimages")
lines!(ax, xs, phi.(sqrt.(xs)) ./ (2 .* sqrt.(xs)); color = PALETTE.red, linestyle = :dash,
       label = L"\text{branch } u = +\sqrt{x} \text{ only}")
xlims!(ax, 0, 5); ylims!(ax, 0, 1.5)
axislegend(ax; position = :rt)
save_fig(fig, @__DIR__, "one-dim")

# ------------------------------------------------------------------- coupling layers
s1(a) = 0.5 * tanh(a); t1(a) = 0.6 * a^2 - 0.6
s2(b) = -0.3 * tanh(b); t2(b) = 0.9 * sin(1.2 * b)
layer1(z1, z2) = (z1, z2 .* exp.(s1.(z1)) .+ t1.(z1))
layer2(z1, z2) = (z1 .* exp.(s2.(z2)) .+ t2.(z2), z2)
function apply_stage(z1, z2, stage)
    z = (z1, z2)
    stage >= 1 && (z = layer1(z...))
    stage >= 2 && (z = layer2(z...))
    return z
end
function logp_stage(stage, y1, y2)
    ld = 0.0
    if stage >= 2
        y1 = (y1 - t2(y2)) * exp(-s2(y2)); ld -= s2(y2)
    end
    if stage >= 1
        ld -= s1(y1); y2 = (y2 - t1(y1)) * exp(-s1(y1))
    end
    return logpdf(Normal(), y1) + logpdf(Normal(), y2) + ld
end

u1 = randn(rng, 2000); u2 = randn(rng, 2000)
g = range(-4, 4, length = 301)
tt = collect(range(-2.6, 2.6, length = 200))
titles = [L"\text{base } \mathcal{N}(0, I_2)", "after coupling layer 1", "after coupling layer 2"]
fig2 = Figure(size = (880, 320))
for stage in 0:2
    ax = Axis(fig2[1, stage + 1]; aspect = DataAspect(), xlabel = L"x_1", ylabel = stage == 0 ? L"x_2" : "",
              title = titles[stage + 1], xgridvisible = false, ygridvisible = false)
    for c in -2.0:1.0:2.0
        for (a, b) in ((fill(c, 200), tt), (tt, fill(c, 200)))
            z = apply_stage(a, b, stage)
            lines!(ax, z[1], z[2]; color = (PALETTE.grey, 0.8), linewidth = 0.7)
        end
    end
    z = apply_stage(u1, u2, stage)
    scatter!(ax, z[1], z[2]; color = (PALETTE.blue, 0.35), markersize = 2.5)
    dens = [exp(logp_stage(stage, a, b)) for a in g, b in g]
    contour!(ax, g, g, dens; levels = [0.01, 0.04, 0.1], color = PALETTE.red, linewidth = 1.2)
    xlims!(ax, -3.6, 3.6); ylims!(ax, -3.6, 4.1)
    stage > 0 && hideydecorations!(ax; ticks = false)
end
save_fig(fig2, @__DIR__, "coupling")
