# Schur complements in Gaussian conditioning and partial correlation.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random, Statistics

set_blog_theme!()

# ---- Figure 1 ------------------------------------------------------------------------------
s1, s2, rho = 1.0, 1.5, 0.7
S = [s1^2 rho*s1*s2; rho*s1*s2 s2^2]
Si = inv(S)
xs2 = range(-4.5, 4.5; length = 300); xs1 = range(-3.2, 3.2; length = 300)
Q = [Si[1, 1] * a^2 + 2 * Si[1, 2] * a * c + Si[2, 2] * c^2 for c in xs2, a in xs1]
b = S[1, 2] / S[2, 2]
cvar = S[1, 1] - S[1, 2]^2 / S[2, 2]          # Schur complement
t = collect(range(-3.2, 3.2; length = 400))

fig = Figure(size = (660, 420))
ax = Axis(fig[2, 1]; xlabel = L"x_2", ylabel = L"x_1")
contour!(ax, xs2, xs1, Q; levels = [0.5, 2, 4.5, 8], color = PALETTE.grey, linewidth = 0.9)
lm = lines!(ax, [-4.5, 4.5], b .* [-4.5, 4.5]; color = PALETTE.red, linestyle = :dash, linewidth = 1.4)
lc = Ref{Any}(nothing)
for x2 in (-2.5, 0.0, 2.5)
    dens = pdf.(Normal(b * x2, sqrt(cvar)), t)
    vlines!(ax, [x2]; color = PALETTE.light, linewidth = 0.8)
    band!(ax, Point2f.(fill(x2, length(t)), t), Point2f.(x2 .+ 1.2 .* dens, t); color = (PALETTE.blue, 0.18))
    lc[] = lines!(ax, x2 .+ 1.2 .* dens, t; color = PALETTE.blue, linewidth = 1.8)
end
lmg = lines!(ax, -4.5 .+ 1.2 .* pdf.(Normal(0, s1), t), t; color = PALETTE.gold, linewidth = 1.8)
xlims!(ax, -4.5, 4.5); ylims!(ax, -3.2, 3.2)
Legend(fig[1, 1], [lm, lc[], lmg],
       [L"\mathbb{E}[X_1 \mid X_2 = x_2] = \Sigma_{12}\Sigma_{22}^{-1}x_2",
        L"X_1 \mid X_2 = x_2,\ \text{variance } \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}",
        L"\text{marginal law of } X_1"];
       orientation = :horizontal, nbanks = 2, labelsize = 12, tellheight = true, halign = :left)
save_fig(fig, @__DIR__, "gaussian-conditioning")

# ---- Figure 2 ------------------------------------------------------------------------------
rng = MersenneTwister(7)
n = 600
Om = [2.0 0.0 -1.2; 0.0 2.0 -1.2; -1.2 -1.2 2.0]
X = rand(rng, MvNormal(zeros(3), Symmetric(inv(Om))), n)'     # n x 3
x1, x2, x3 = X[:, 1], X[:, 2], X[:, 3]
r1 = x1 .- (dot(x1, x3) / dot(x3, x3)) .* x3
r2 = x2 .- (dot(x2, x3) / dot(x3, x3)) .* x3
lim = 1.05 * maximum(abs.(vcat(x1, x2)))
fmt(v) = string(round(v; digits = 2))

fig2 = Figure(size = (760, 360))
a1 = Axis(fig2[1, 1]; xlabel = L"x_1", ylabel = L"x_2", aspect = DataAspect(),
          title = L"\text{(a) } X_1 \text{ vs } X_2, \text{ sample corr. } %$(fmt(cor(x1, x2)))")
sc = scatter!(a1, x1, x2; color = x3, colormap = Reverse(:RdBu), colorrange = (-3, 3), markersize = 5)
a2 = Axis(fig2[1, 2]; xlabel = L"\text{residual of } x_1", ylabel = L"\text{residual of } x_2", aspect = DataAspect(),
          title = L"\text{(b) residuals on } X_3, \text{ corr. } %$(fmt(cor(r1, r2)))")
scatter!(a2, r1, r2; color = (PALETTE.blue, 0.5), markersize = 5)
for a in (a1, a2)
    xlims!(a, -lim, lim); ylims!(a, -lim, lim)
end
Colorbar(fig2[1, 3], sc; label = L"x_3", height = Relative(0.8))
save_fig(fig2, @__DIR__, "partial-correlation")
