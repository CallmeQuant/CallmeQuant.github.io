# Box-Muller map and Marsaglia's polar method.
# Packages: CairoMakie, Distributions, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(1958)
box_muller(u1, u2) = (sqrt(-2log(u1)) * cos(2pi * u2), sqrt(-2log(u1)) * sin(2pi * u2))

# ---------- Figure 1: the map acting on a grid ----------
lv1 = [0.02, 0.1, 0.3, 0.6, 0.9]
lv2 = (0:7) ./ 8
n_pts = 1500
u1 = 1 .- rand(rng, n_pts); u2 = rand(rng, n_pts)
zs = box_muller.(u1, u2)
z1 = first.(zs); z2 = last.(zs)
tt = range(0, 2pi; length = 400)

fig = Figure(size = (800, 380))
ax1 = Axis(fig[1, 1]; xlabel = L"u_1", ylabel = L"u_2", title = L"\text{Uniform pairs } (U_1, U_2)",
           aspect = DataAspect(), xgridvisible = false, ygridvisible = false)
scatter!(ax1, u1, u2; markersize = 2.5, color = (PALETTE.grey, 0.55))
hlines!(ax1, collect(lv2); color = (PALETTE.ink, 0.6), linewidth = 0.6)
for (k, a) in enumerate(lv1)
    vlines!(ax1, [a]; color = CYCLE[k], linewidth = 1.8)
end
xlims!(ax1, 0, 1); ylims!(ax1, 0, 1)

ax2 = Axis(fig[1, 2]; xlabel = L"z_1", ylabel = L"z_2", title = L"\text{Image } (Z_1, Z_2)",
           aspect = DataAspect(), xgridvisible = false, ygridvisible = false)
scatter!(ax2, z1, z2; markersize = 2.5, color = (PALETTE.grey, 0.55))
for b in lv2
    lines!(ax2, [0, 3.3cos(2pi * b)], [0, 3.3sin(2pi * b)]; color = (PALETTE.ink, 0.6), linewidth = 0.6)
end
for (k, a) in enumerate(lv1)
    r = sqrt(-2log(a))
    lines!(ax2, r .* cos.(tt), r .* sin.(tt); color = CYCLE[k], linewidth = 1.8, label = "u₁ = $a")
end
xlims!(ax2, -3.4, 3.4); ylims!(ax2, -3.4, 3.4)
Legend(fig[1, 3], ax2; framevisible = false)
save_fig(fig, @__DIR__, "box-muller-map")

# ---------- Figure 2: polar method ----------
m = 1200
v = 2 .* rand(rng, m, 2) .- 1
s = vec(sum(v .^ 2; dims = 2))
acc = (s .> 0) .& (s .< 1)
N = 200_000
w = 2 .* rand(rng, round(Int, 1.4N), 2) .- 1
sw = vec(sum(w .^ 2; dims = 2))
keep = findall((sw .> 0) .& (sw .< 1))[1:N]
w = w[keep, :]; sw = sw[keep]
zz = w .* sqrt.(-2 .* log.(sw) ./ sw)

fig = Figure(size = (880, 310))
a1 = Axis(fig[1, 1]; xlabel = L"v_1", ylabel = L"v_2", aspect = DataAspect(),
          title = @sprintf("Accept rate %.3f (π/4 ≈ 0.785)", mean(acc)), titlesize = 13,
          xgridvisible = false, ygridvisible = false, xticks = [-1, 0, 1], yticks = [-1, 0, 1])
scatter!(a1, v[acc, 1], v[acc, 2]; markersize = 3, color = PALETTE.blue)
scatter!(a1, v[.!acc, 1], v[.!acc, 2]; markersize = 3, color = PALETTE.red)
lines!(a1, cos.(tt), sin.(tt); color = PALETTE.ink, linewidth = 1)
xlims!(a1, -1, 1); ylims!(a1, -1, 1)

a2 = Axis(fig[1, 2]; xlabel = L"S = V_1^2 + V_2^2 \text{ (accepted)}", ylabel = "density",
          title = L"S \sim \mathrm{Uniform}(0,1)")
hist!(a2, sw; bins = range(0, 1; length = 41), normalization = :pdf,
      color = (PALETTE.grey, 0.45), strokecolor = :white, strokewidth = 0.5)
hlines!(a2, [1.0]; color = PALETTE.ink, linewidth = 1.4)
xlims!(a2, 0, 1); ylims!(a2, 0, 1.35)

a3 = Axis(fig[1, 3]; xlabel = L"Z_1", ylabel = "density", title = L"\text{Output vs. } N(0,1) \text{ density}")
hist!(a3, zz[:, 1]; bins = range(-4, 4; length = 61), normalization = :pdf,
      color = (PALETTE.grey, 0.45), strokecolor = :white, strokewidth = 0.5)
x = range(-4, 4; length = 400)
lines!(a3, x, pdf.(Normal(), x); color = PALETTE.blue)
xlims!(a3, -4, 4); ylims!(a3, 0, 0.47)
colsize!(fig.layout, 1, Relative(0.28))
save_fig(fig, @__DIR__, "polar-method")
