# Ridge shrinkage: ridge path geometry for two correlated covariates; shrinkage factors and MSE.
# Packages: CairoMakie, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra

set_blog_theme!()

# ---- Figure 1 ------------------------------------------------------------------------------
rho = 0.9
G = [1.0 rho; rho 1.0]
b_ols = [2.0, -0.3]
ridge(l) = (G + l * I) \ (G * b_ols)
lams = vcat(0.0, 10 .^ range(-3, 2.5; length = 400))
path = reduce(hcat, ridge.(lams))

fig = Figure(size = (760, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"\beta_1", ylabel = L"\beta_2", aspect = DataAspect(),
           title = "(a) Ridge path in the parameter plane")
b1 = range(-1.2, 3.0; length = 250); b2 = range(-1.2, 2.2; length = 250)
rss = [dot([x - b_ols[1], y - b_ols[2]], G * [x - b_ols[1], y - b_ols[2]]) for x in b1, y in b2]
t = range(0, 2pi; length = 200)
for r in (0.5, 1.0, 1.5)
    lines!(ax1, r .* cos.(t), r .* sin.(t); color = PALETTE.light, linewidth = 1)
end
contour!(ax1, b1, b2, rss; levels = [0.05, 0.2, 0.5, 1.0, 1.6, 2.4], color = PALETTE.grey, linewidth = 0.9)
for (v, lab, dy) in (([1, 1] ./ sqrt(2), L"v_1", 0.08), ([1, -1] ./ sqrt(2), L"v_2", -0.15))
    arrows!(ax1, [0.0], [0.0], [0.9 * v[1]], [0.9 * v[2]]; color = PALETTE.green, arrowsize = 9)
    text!(ax1, 0.9 * v[1] + 0.03, 0.9 * v[2] + dy; text = lab, color = PALETTE.green)
end
lines!(ax1, path[1, :], path[2, :]; color = PALETTE.blue, linewidth = 2.4)
for l in (0.1, 1.0, 5.0)
    q = ridge(l)
    scatter!(ax1, [q[1]], [q[2]]; color = PALETTE.blue, markersize = 7)
    text!(ax1, q[1] + 0.04, q[2] - 0.06; text = L"\lambda = %$(l)", color = PALETTE.blue,
          fontsize = 12, align = (:left, :top))
end
scatter!(ax1, [b_ols[1]], [b_ols[2]]; color = PALETTE.red, markersize = 11)
text!(ax1, b_ols[1] + 0.1, b_ols[2]; text = L"\hat\beta^{\mathrm{OLS}}", color = PALETTE.red, align = (:left, :center))
scatter!(ax1, [0.0], [0.0]; marker = :cross, color = PALETTE.ink, markersize = 10)
xlims!(ax1, -1.2, 3.0); ylims!(ax1, -1.2, 2.2)

ax2 = Axis(fig[1, 2]; xscale = log10, xlabel = L"\lambda \text{ (log scale)}",
           title = "(b) Coefficients along the path")
lp = 10 .^ range(-3, 2.5; length = 400)
pp = reduce(hcat, ridge.(lp))
hlines!(ax2, [0.0]; color = PALETTE.ink, linewidth = 0.7)
lines!(ax2, lp, pp[1, :]; color = CYCLE[1], label = L"\hat\beta_{\lambda,1}")
lines!(ax2, lp, pp[2, :]; color = CYCLE[2], label = L"\hat\beta_{\lambda,2}")
axislegend(ax2; position = :rt)
colsize!(fig.layout, 2, Relative(0.53))
save_fig(fig, @__DIR__, "ridge-geometry")

# ---- Figure 2 ------------------------------------------------------------------------------
d2 = [50.0, 20.0, 8.0, 3.0, 1.0, 0.3]
alpha = [1.0, -0.8, 0.6, 0.5, -0.4, 0.3]
sigma2 = 1.0
lam = 10 .^ range(-3, 3; length = 400)
S = [d / (d + l) for l in lam, d in d2]
bias2 = vec(sum((1 .- S) .^ 2 .* (alpha .^ 2)'; dims = 2))
vari = vec(sum(S .^ 2 .* (sigma2 ./ d2)'; dims = 2))
mse = bias2 .+ vari
mse_ols = sum(sigma2 ./ d2)
lam_hk = 2 * sigma2 / maximum(alpha .^ 2)

fig2 = Figure(size = (760, 330))
bx1 = Axis(fig2[1, 1]; xscale = log10, xlabel = L"\lambda \text{ (log scale)}",
           ylabel = L"s_j(\lambda) = d_j^2/(d_j^2+\lambda)", title = "(a) Shrinkage factors")
for j in eachindex(d2)
    lines!(bx1, lam, S[:, j]; color = CYCLE[j], label = L"d_{%$j}^2 = %$(d2[j])")
end
ylims!(bx1, 0, 1.02)
axislegend(bx1; position = :lb, labelsize = 11, rowgap = 0)

bx2 = Axis(fig2[1, 2]; xscale = log10, xlabel = L"\lambda \text{ (log scale)}",
           title = L"\text{(b) Mean squared error of } \hat\beta_\lambda")
vspan!(bx2, [lam[1]], [lam_hk]; color = (PALETTE.light, 0.7))
hlines!(bx2, [mse_ols]; color = PALETTE.grey, linestyle = :dot, linewidth = 1.2)
text!(bx2, lam[end], mse_ols + 0.05; text = "OLS", color = PALETTE.grey, align = (:right, :bottom), fontsize = 12)
lines!(bx2, lam, mse; color = PALETTE.ink, label = "MSE")
lines!(bx2, lam, bias2; color = PALETTE.red, linestyle = :dash, label = "squared bias")
lines!(bx2, lam, vari; color = PALETTE.blue, linestyle = :dash, label = "variance")
k = argmin(mse)
scatter!(bx2, [lam[k]], [mse[k]]; color = PALETTE.ink, markersize = 7)
ylims!(bx2, 0, 5.6)
axislegend(bx2; position = (1.0, 0.82))
save_fig(fig2, @__DIR__, "shrinkage-mse")
