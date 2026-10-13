# Geometry of high-dimensional balls and Gaussians.
# Packages: CairoMakie, Distributions, Random, LinearAlgebra, Statistics, SpecialFunctions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, LinearAlgebra, Statistics, SpecialFunctions

set_blog_theme!()
rng = MersenneTwister(1961)

fig = Figure(size = (1000, 320))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = L"\text{dimension } d", title = "Volume of the unit ball")
d = 1:50
logV = d ./ 2 .* log(pi) .- loggamma.(d ./ 2 .+ 1)
scatterlines!(ax1, d, exp.(logV); color = PALETTE.blue, markersize = 5, linewidth = 1, label = L"V_d \text{ (unit ball)}")
scatterlines!(ax1, d, exp.(logV .- d .* log(2)); color = PALETTE.red, markersize = 5, linewidth = 1,
              label = L"V_d/2^d \text{ (ball / cube)}")
ylims!(ax1, 1e-28, 50)
axislegend(ax1; position = :lb)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{relative shell thickness } \varepsilon",
           title = L"\text{Volume fraction with } \Vert x\Vert \geq 1 - \varepsilon")
eps = range(0, 0.5; length = 300)
for (j, dd) in enumerate([2, 10, 100, 1000])
    lines!(ax2, eps, 1 .- (1 .- eps) .^ dd; color = CYCLE[j], label = L"d = %$dd")
end
xlims!(ax2, 0, 0.5); ylims!(ax2, 0, 1.03)
axislegend(ax2; position = :rb)

ax3 = Axis(fig[1, 3]; xlabel = L"\Vert Z\Vert/\sqrt{d}", title = L"\text{Radius of } Z \sim N(0, I_d)\text{, rescaled}")
s = range(0.001, 2.2; length = 800)
for (j, dd) in enumerate([1, 3, 10, 100, 1000])
    lines!(ax3, s, pdf.(Chi(dd), s .* sqrt(dd)) .* sqrt(dd); color = CYCLE[j], label = L"d = %$dd")
end
xlims!(ax3, 0, 2.2); ylims!(ax3, 0, 19.5)
axislegend(ax3; position = :rt)
save_fig(fig, @__DIR__, "volumes-and-shells")

fig = Figure(size = (900, 320))
bx1 = Axis(fig[1, 1]; xlabel = "angle between two independent Gaussian vectors (degrees)",
           ylabel = "density (per degree)",
           title = "Random directions are nearly orthogonal", xticks = 0:45:180)
for (j, dd) in enumerate([3, 30, 300])
    Z1 = randn(rng, 20000, dd); Z2 = randn(rng, 20000, dd)
    c = vec(sum(Z1 .* Z2; dims = 2)) ./ (norm.(eachrow(Z1)) .* norm.(eachrow(Z2)))
    hist!(bx1, rad2deg.(acos.(clamp.(c, -1, 1))); bins = range(0, 180; length = 91), normalization = :pdf,
          color = (CYCLE[j], 0.5), label = L"d = %$dd")
end
vlines!(bx1, [90]; color = PALETTE.ink, linestyle = :dot)
xlims!(bx1, 0, 180)
axislegend(bx1; position = :lt)

pdists(X) = [norm(X[i, :] .- X[j, :]) for i in 1:size(X, 1) for j in i+1:size(X, 1)]
dims = unique(round.(Int, 10 .^ range(0, 4; length = 25)))
med = Float64[]; lo = Float64[]; hi = Float64[]
for dd in dims
    v = [(D = pdists(randn(rng, 100, dd)); (maximum(D) - minimum(D)) / minimum(D)) for _ in 1:10]
    push!(med, median(v)); push!(lo, minimum(v)); push!(hi, maximum(v))
end
bx2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{dimension } d",
           ylabel = L"(D_{\max} - D_{\min})/D_{\min}", title = "Pairwise distances of 100 Gaussian points")
band!(bx2, dims, lo, hi; color = PALETTE.light)
scatterlines!(bx2, dims, med; color = PALETTE.blue, markersize = 5, linewidth = 1, label = "median of 10 replicates")
lines!(bx2, dims, 8 ./ sqrt.(dims); color = PALETTE.ink, linestyle = :dash, linewidth = 1, label = L"\text{slope } d^{-1/2}")
axislegend(bx2; position = :rt)
save_fig(fig, @__DIR__, "angles-and-distances")
