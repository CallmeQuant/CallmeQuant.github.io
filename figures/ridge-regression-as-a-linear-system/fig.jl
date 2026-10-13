# Ridge regression with p > n: coefficient paths and the leave-one-out shortcut.
# Packages: CairoMakie, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1)
n, p, rho = 30, 60, 0.6
Sigma = [rho^abs(i - j) for i in 1:p, j in 1:p]
X = randn(rng, n, p) * cholesky(Symmetric(Sigma)).U
beta = vcat([2.0, -1.5, 1.0, 0.8, -0.6], zeros(p - 5))
y = X * beta .+ 2.0 .* randn(rng, n)

F = svd(X)
U, d, V = F.U, F.S, F.V
Uty = U' * y
lams = 10 .^ range(-3, 4; length = 200)
paths = reduce(hcat, [V * (d ./ (d .^ 2 .+ l) .* Uty) for l in lams])   # p x length(lams)
bmin = V * (Uty ./ d)

hatm(l) = U * Diagonal(d .^ 2 ./ (d .^ 2 .+ l)) * U'
function loo_short(l)
    H = hatm(l); r = y - H * y
    mean((r ./ (1 .- diag(H))) .^ 2)
end
function gcv(l)
    H = hatm(l); r = y - H * y
    mean(r .^ 2) / (1 - tr(H) / n)^2
end
function loo_brute(l)
    e = map(1:n) do i
        m = setdiff(1:n, i)
        b = (X[m, :]' * X[m, :] + l * I) \ (X[m, :]' * y[m])
        y[i] - dot(X[i, :], b)
    end
    mean(e .^ 2)
end

fig = Figure(size = (760, 340))
ax1 = Axis(fig[1, 1]; xscale = log10, xlabel = L"\lambda \text{ (log scale)}",
           ylabel = L"\hat\beta_{\lambda,j}", title = L"\text{(a) Paths; triangles mark } (X^{+}y)_j")
for j in 6:p
    lines!(ax1, lams, paths[j, :]; color = PALETTE.light, linewidth = 0.8)
end
for j in 1:5
    lines!(ax1, lams, paths[j, :]; color = CYCLE[j], linewidth = 1.8)
    scatter!(ax1, [lams[1] * 0.55], [bmin[j]]; marker = :ltriangle, markersize = 12, color = CYCLE[j])
end
hlines!(ax1, [0.0]; color = PALETTE.ink, linewidth = 0.7)
xlims!(ax1, lams[1] * 0.45, lams[end])

ax2 = Axis(fig[1, 2]; xscale = log10, xlabel = L"\lambda \text{ (log scale)}",
           ylabel = "prediction error estimate", title = "(b) Leave-one-out error")
lg = 10 .^ range(-2, 3; length = 80)
lb = 10 .^ range(-2, 3; length = 11)
lines!(ax2, lg, loo_short.(lg); color = PALETTE.blue, label = "LOO (shortcut)")
scatter!(ax2, lb, loo_brute.(lb); color = :white, strokecolor = PALETTE.red, strokewidth = 1.6,
         markersize = 9, label = L"\text{LOO } (n \text{ refits})")
lines!(ax2, lg, gcv.(lg); color = PALETTE.gold, linestyle = :dash, label = "GCV")
axislegend(ax2; position = :lb)
save_fig(fig, @__DIR__, "paths-and-loo")
