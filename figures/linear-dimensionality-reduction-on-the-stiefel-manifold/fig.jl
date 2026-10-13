# Trace-ratio objective maximized by Riemannian gradient ascent on the Stiefel manifold,
# compared with PCA and with the generalized-eigenvector (ratio-trace) solution.
# Packages: CairoMakie, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Printf, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(0)
d, r, nper = 6, 2, 100
A = randn(rng, d, d) * Diagonal([3.0, 2.5, 2.0, 1.0, 0.6, 0.4])
means = randn(rng, 3, d)
X = vcat([randn(rng, nper, d) * A' .+ means[c, :]' for c in 1:3]...)
lab = repeat(1:3; inner = nper)
mu = vec(mean(X; dims = 1))
centred(Z) = Z .- mean(Z; dims = 1)
Sw = sum(centred(X[lab .== c, :])' * centred(X[lab .== c, :]) for c in 1:3) / size(X, 1)
Sb = sum(mean(lab .== c) * (vec(mean(X[lab .== c, :]; dims = 1)) - mu) * (vec(mean(X[lab .== c, :]; dims = 1)) - mu)' for c in 1:3)
St = centred(X)' * centred(X) / size(X, 1)

f(M) = tr(M' * Sb * M) / tr(M' * Sw * M)
function qf(B)
    F = qr(B); Q = Matrix(F.Q)[:, 1:size(B, 2)]
    return Q * Diagonal(sign.(diag(F.R)))
end
tangent(M, G) = (S = M' * G; G - M * (S + S') / 2)
function egrad(M)
    a, b = tr(M' * Sb * M), tr(M' * Sw * M)
    return 2 * Sb * M / b - 2 * a / b^2 * Sw * M
end
function ascend(M, iters)
    vals = [f(M)]
    for _ in 1:iters
        Z = tangent(M, egrad(M)); t = 1.0
        while f(qf(M + t * Z)) < f(M) + 1e-4 * t * sum(abs2, Z) && t > 1e-10
            t /= 2
        end
        M = qf(M + t * Z); push!(vals, f(M))
    end
    return M, vals
end

M_pca = eigen(Symmetric(St)).vectors[:, end:-1:end-r+1]
M_gev = qf(eigen(Symmetric(Sb), Symmetric(Sw)).vectors[:, end:-1:end-r+1])
runs = [ascend(qf(randn(rng, d, r)), 600) for _ in 1:5]
M_opt = runs[argmax([last(o[2]) for o in runs])][1]
lo, hi = 0.0, 50.0
for _ in 1:100
    rho = (lo + hi) / 2
    if sum(eigvals(Symmetric(Sb - rho * Sw))[end-r+1:end]) > 0
        global lo = rho
    else
        global hi = rho
    end
end
rho_star = (lo + hi) / 2

fig = Figure(size = (720, 660))
panels = [("(a) PCA", M_pca), ("(b) ratio-trace optimum (generalized eigenvectors)", M_gev),
          ("(c) trace-ratio optimum (Riemannian ascent)", M_opt)]
for (i, (title, M)) in enumerate(panels)
    ax = Axis(fig[fldmod1(i, 2)...]; xlabel = L"y_1", ylabel = L"y_2", titlesize = 12,
              title = title * @sprintf("\nf(M) = %.2f", f(M)))
    Y = (X .- mu') * M
    for c in 1:3
        scatter!(ax, Y[lab .== c, 1], Y[lab .== c, 2]; color = (CYCLE[c], 0.7), markersize = 5)
    end
end
ax = Axis(fig[2, 2]; xscale = log10, xlabel = L"\text{iteration } k + 1 \text{ (log scale)}", ylabel = L"f(M_k)",
          title = "(d) Objective along the iterations", titlesize = 12)
for (i, (_, v)) in enumerate(runs)
    lines!(ax, 1:length(v), v; color = (PALETTE.blue, 0.8), linewidth = 1.3,
           label = i == 1 ? "Riemannian ascent, 5 starts" : nothing)
end
hlines!(ax, [rho_star]; color = PALETTE.ink, linewidth = 1, label = L"\text{global maximum } \rho_\star")
hlines!(ax, [f(M_gev)]; color = PALETTE.gold, linestyle = :dash, linewidth = 1.4, label = "ratio-trace optimum")
hlines!(ax, [f(M_pca)]; color = PALETTE.grey, linestyle = :dot, linewidth = 1.4, label = "PCA")
axislegend(ax; position = (0.02, 0.62), labelsize = 10)
save_fig(fig, @__DIR__, "trace-ratio-projections")
