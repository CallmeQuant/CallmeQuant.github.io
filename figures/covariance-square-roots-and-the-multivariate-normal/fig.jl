# Three square roots of the same covariance matrix applied to the same standard normal draws.
# Packages: CairoMakie, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random

set_blog_theme!()
rng = MersenneTwister(1729)
Sigma = [2.0 1.2; 1.2 1.0]
E = eigen(Symmetric(Sigma))
lam = reverse(E.values); U = E.vectors[:, [2, 1]]       # decreasing order
U[1, 1] < 0 && (U[:, 1] .*= -1)
U[2, 2] < 0 && (U[:, 2] .*= -1)
factors = [
    (L"\text{(a) } Z \sim N(0, I_2)", Matrix(1.0I, 2, 2)),
    (L"\text{(b) Cholesky: } LZ", Matrix(cholesky(Symmetric(Sigma)).L)),
    (L"\text{(c) symmetric root: } \Sigma^{1/2}Z", U * Diagonal(sqrt.(lam)) * U'),
    (L"\text{(d) eigen factor: } U\Lambda^{1/2}Z", U * Diagonal(sqrt.(lam))),
]
n = 400
Z = randn(rng, 2, n)
ang = atan.(Z[2, :], Z[1, :])
t = range(0, 2pi; length = 300)
circ = vcat(cos.(t)', sin.(t)')

fig = Figure(size = (640, 640))
axs = Axis[]
for (k, (title, A)) in enumerate(factors)
    i, j = fldmod1(k, 2)
    ax = Axis(fig[i, j]; title = title, aspect = DataAspect(),
              xlabel = i == 2 ? L"x_1" : "", ylabel = j == 1 ? L"x_2" : "")
    X = A * Z
    scatter!(ax, X[1, :], X[2, :]; color = ang, colormap = :twilight, markersize = 5)
    for r in (1, 2)
        El = A * (r .* circ)
        lines!(ax, El[1, :], El[2, :]; color = PALETTE.ink, linewidth = 1)
    end
    for (e, col) in (([1.0, 0.0], PALETTE.red), ([0.0, 1.0], PALETTE.green))
        v = 2 .* (A * e)
        arrows!(ax, [0.0], [0.0], [v[1]], [v[2]]; color = col, linewidth = 2.2, arrowsize = 12)
    end
    xlims!(ax, -3.8, 3.8); ylims!(ax, -3.8, 3.8)
    push!(axs, ax)
end
linkaxes!(axs...)
hidexdecorations!(axs[1]; grid = false); hidexdecorations!(axs[2]; grid = false)
hideydecorations!(axs[2]; grid = false); hideydecorations!(axs[4]; grid = false)
save_fig(fig, @__DIR__, "three-square-roots")
