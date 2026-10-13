# Cholesky factors of the covariance and of the precision matrix of a stationary AR(1) vector,
# and draws obtained from the precision factor by back-substitution (note: sampling the multivariate normal).
# Packages: CairoMakie, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random

set_blog_theme!()
const phi = 0.9
ar1_cov(d) = [phi^abs(i - j) / (1 - phi^2) for i in 1:d, j in 1:d]
function ar1_prec(d)
    dv = vcat(1.0, fill(1 + phi^2, d - 2), 1.0)
    return SymTridiagonal(dv, fill(-phi, d - 1))
end

d_small = 30
L_S = Matrix(cholesky(Symmetric(ar1_cov(d_small))).L)
L_Q = Matrix(cholesky(Matrix(ar1_prec(d_small))).L)
cmap = cgrad([colorant"#ffffff", PALETTE.blue, PALETTE.ink])

fig = Figure(size = (860, 310))
for (k, (M, title)) in enumerate([(L_S, L"\text{(a) } L_\Sigma"), (L_Q, L"\text{(b) } L_Q,\ Q = \Sigma^{-1}")])
    ax = Axis(fig[1, k]; title = title, aspect = DataAspect(), yreversed = true,
              xticks = [1, d_small], yticks = [1, d_small], xgridvisible = false, ygridvisible = false)
    A = [abs(x) < 1e-12 ? NaN : abs(x) for x in M]'   # heatmap!(x, y, A) uses A[x, y]
    heatmap!(ax, 1:d_small, 1:d_small, A; colormap = cmap, colorrange = (0, maximum(abs.(M))),
             nan_color = :white)
    text!(ax, d_small + 0.4, 0.6; text = "$(count(x -> abs(x) > 1e-12, M)) nonzeros",
          align = (:right, :top), fontsize = 12)
    limits!(ax, 0.5, d_small + 0.5, d_small + 0.5, 0.5)
end

d = 300
U = UpperTriangular(Matrix(cholesky(Matrix(ar1_prec(d))).U))   # equals L_Q^T
rng = MersenneTwister(7)
sdv = 1 / sqrt(1 - phi^2)
ax3 = Axis(fig[1, 3]; xlabel = L"\text{coordinate } i", title = L"\text{(c) draws } y = L_Q^{-\top} z,\ d = 300")
band!(ax3, [1.0, d], [-2sdv, -2sdv], [2sdv, 2sdv]; color = PALETTE.light)
for j in 1:3
    z = randn(rng, d)
    lines!(ax3, 1:d, U \ z; color = CYCLE[j], linewidth = 1.2)   # back-substitution
end
xlims!(ax3, 1, d); ylims!(ax3, -8.5, 8.5)
colsize!(fig.layout, 3, Relative(0.5))
save_fig(fig, @__DIR__, "precision-cholesky")
