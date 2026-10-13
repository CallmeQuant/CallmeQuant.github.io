# Accuracy of least squares via the normal equations (Cholesky), Householder QR and the SVD.
# Packages: CairoMakie, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(31)
n, p, reps = 100, 10, 20
kappas = 10 .^ range(0, 9; length = 19)
u = eps(Float64) / 2

function solve_normal(A, y)
    C = cholesky(Symmetric(A' * A); check = false)
    issuccess(C) || return fill(NaN, size(A, 2))
    return C \ (A' * y)
end
solve_qr(A, y) = qr(A) \ y                      # Householder QR (LAPACK geqrf)
function solve_svd(A, y)
    F = svd(A)
    return F.V * ((F.U' * y) ./ F.S)
end

methods = [("normal equations + Cholesky", solve_normal, PALETTE.red, :circle),
           ("Householder QR", solve_qr, PALETTE.blue, :rect),
           ("SVD", solve_svd, PALETTE.green, :utriangle)]
med = Dict(m[1] => fill(NaN, length(kappas)) for m in methods)
for (i, k) in enumerate(kappas)
    errs = Dict(m[1] => Float64[] for m in methods)
    for r in 1:reps
        U = Matrix(qr(randn(rng, n, p)).Q)[:, 1:p]
        V = Matrix(qr(randn(rng, p, p)).Q)
        A = U * Diagonal(10 .^ range(0, -log10(k); length = p)) * V'
        beta = normalize(randn(rng, p))
        y = A * beta
        for (name, f, _, _) in methods
            b = f(A, y)
            push!(errs[name], all(isfinite, b) ? norm(b - beta) : NaN)
        end
    end
    for (name, _, _, _) in methods
        e = errs[name]
        med[name][i] = any(isnan, e) ? NaN : min(median(e), 1.0)
    end
end

fig = Figure(size = (640, 390))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
          xlabel = L"\text{condition number } \kappa_2(A) = \sigma_1/\sigma_p",
          ylabel = L"\text{relative error } \Vert\hat\beta - \beta\Vert / \Vert\beta\Vert")
lines!(ax, kappas, u .* kappas; color = PALETTE.grey, linestyle = :dash, linewidth = 1)
k2 = kappas[u .* kappas .^ 2 .< 3]
lines!(ax, k2, u .* k2 .^ 2; color = PALETTE.grey, linestyle = :dot, linewidth = 1.3)
text!(ax, 1e6, u * 1e6 * 5; text = L"u\,\kappa", color = PALETTE.grey, align = (:right, :bottom))
text!(ax, 3e5, u * 9e10 * 3; text = L"u\,\kappa^2", color = PALETTE.grey, align = (:right, :bottom))
for (name, _, col, mk) in methods
    ok = .!isnan.(med[name])
    scatterlines!(ax, kappas[ok], med[name][ok]; color = col, marker = mk, markersize = 9, label = name)
end
ylims!(ax, 1e-17, 3)
axislegend(ax; position = :lt)
save_fig(fig, @__DIR__, "least-squares-accuracy")
