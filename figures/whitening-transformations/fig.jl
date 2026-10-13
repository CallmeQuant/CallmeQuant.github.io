# Whitening a uniform parallelogram: ZCA, PCA and Cholesky whitening.
# Packages: CairoMakie, LaTeXStrings, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LaTeXStrings, LinearAlgebra, Printf, Random

set_blog_theme!()
rng = MersenneTwister(404)
n = 1500
S = (2 .* rand(rng, 2, n) .- 1) .* sqrt(3)          # uniform on a square, identity covariance
M = [1.2 0.7; 0.25 0.6]
X = M * S
Sigma = M * M'
E = eigen(Symmetric(Sigma))
lam = reverse(E.values); U = E.vectors[:, [2, 1]]
U = U .* sign.(U[1, :])'
W = [("ZCA", U * Diagonal(lam .^ -0.5) * U'),
     ("PCA", Diagonal(lam .^ -0.5) * U'),
     ("Cholesky", inv(Matrix(cholesky(Symmetric(Sigma)).L)))]
dist(Wm) = 2 + tr(Sigma) - 2 * tr(Wm * Sigma)
sq = [-1 1 1 -1 -1; -1 -1 1 1 -1] .* sqrt(3)
panels = vcat([(L"\text{(a) data } X", Matrix(1.0I, 2, 2))],
              [(latexstring("\\text{($(c)) $(k): } \\mathbb{E}\\Vert Z - X\\Vert^2 = " * @sprintf("%.2f", dist(Wm))), Wm)
               for (c, (k, Wm)) in zip("bcd", W)])

fig = Figure(size = (640, 640))
axs = Axis[]
for (k, (title, Wm)) in enumerate(panels)
    i, j = fldmod1(k, 2)
    ax = Axis(fig[i, j]; title = title, aspect = DataAspect(), titlesize = 13,
              xlabel = i == 2 ? "first coordinate" : "", ylabel = j == 1 ? "second coordinate" : "")
    Y = Wm * X
    scatter!(ax, Y[1, :], Y[2, :]; color = S[1, :], colormap = :viridis, markersize = 3)
    B = Wm * M * sq
    lines!(ax, B[1, :], B[2, :]; color = PALETTE.ink, linewidth = 1.2)
    c = Wm * M * [sqrt(3), sqrt(3)]
    scatter!(ax, [c[1]], [c[2]]; color = PALETTE.red, markersize = 10)
    xlims!(ax, -3.6, 3.6); ylims!(ax, -3.6, 3.6)
    push!(axs, ax)
end
linkaxes!(axs...)
save_fig(fig, @__DIR__, "whitened-parallelogram")
