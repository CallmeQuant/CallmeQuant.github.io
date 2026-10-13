# Kernel ridge regression: the fitted function as a sum of kernel sections at the data, and choice of the
# penalty by the closed-form leave-one-out formula.
# Packages: CairoMakie, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, LinearAlgebra, Statistics

set_blog_theme!()
rng = MersenneTwister(1971)
const WIDTH = 0.08
kern(a, b) = exp.(-0.5 .* (a .- b') .^ 2 ./ WIDTH^2)
f_true(x) = x * sin(3π * x) + 0.3
n = 40
X = sort(rand(rng, n))
y = f_true.(X) .+ 0.2 .* randn(rng, n)
K = kern(X, X)
xs = collect(range(0, 1; length = 500))
Ks = kern(xs, X)
fit(lam) = (K + n * lam * I) \ y
function loo(lam)
    H = K / (K + n * lam * I)
    r = (y - H * y) ./ (1 .- diag(H))
    return mean(r .^ 2), tr(H)
end
function loo_brute(lam)
    e = map(1:n) do i
        keep = setdiff(1:n, i)
        a = (K[keep, keep] + n * lam * I) \ y[keep]
        y[i] - dot(K[i, keep], a)
    end
    return mean(e .^ 2)
end
lams = exp.(range(log(1e-6), log(1.0); length = 120))
cv = [loo(l) for l in lams]
lam_hat = lams[argmin(first.(cv))]
alpha = fit(lam_hat)

fig = Figure(size = (660, 440))
ax1 = Axis(fig[1, 1]; ylabel = L"y")
lines!(ax1, xs, f_true.(xs); color = PALETTE.ink, linestyle = :dash, linewidth = 1.2, label = "true regression function")
lines!(ax1, xs, Ks * alpha; color = PALETTE.blue, linewidth = 2.4, label = L"\hat f = \sum_i \hat\alpha_i k(\cdot, x_i)")
scatter!(ax1, X, y; color = PALETTE.red, markersize = 8, label = "data")
ylims!(ax1, -1.3, 1.6)
axislegend(ax1; position = :lb, orientation = :horizontal, labelsize = 12)
ax2 = Axis(fig[2, 1]; xlabel = L"x", ylabel = L"\hat\alpha_i \, k(x, x_i)")
hlines!(ax2, [0.0]; color = PALETTE.ink, linewidth = 0.8)
for i in 1:n
    linesegments!(ax2, [X[i], X[i]], [0.0, alpha[i]]; color = PALETTE.grey, linewidth = 0.8)
    lines!(ax2, xs, alpha[i] .* Ks[:, i]; color = (CYCLE[mod1(i, 4)], 0.85), linewidth = 1.1)
end
linkxaxes!(ax1, ax2); xlims!(ax2, 0, 1)
hidexdecorations!(ax1; grid = false)
rowsize!(fig.layout, 1, Relative(0.56))
save_fig(fig, @__DIR__, "kernel-sections")

fig = Figure(size = (780, 320))
ax3 = Axis(fig[1, 1]; xlabel = L"x", title = "Fits for three penalties")
for (k, l) in enumerate([1e-6, lam_hat, 0.3])
    lines!(ax3, xs, Ks * fit(l); color = CYCLE[[4, 1, 3][k]], linewidth = 1.8, label = L"\lambda = %$(round(l; sigdigits = 2))")
end
scatter!(ax3, X, y; color = PALETTE.red, markersize = 7)
xlims!(ax3, 0, 1); ylims!(ax3, -1.3, 1.6)
axislegend(ax3; position = :lb, labelsize = 12)
ax4 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{penalty } \lambda",
           ylabel = "leave-one-out mean squared error", title = "Leave-one-out error")
lines!(ax4, lams, first.(cv); color = PALETTE.blue, linewidth = 2.2, label = "closed-form LOO")
chk = lams[1:12:end]
scatter!(ax4, chk, [loo_brute(l) for l in chk]; color = :white, strokecolor = PALETTE.red, strokewidth = 1.5,
         markersize = 9, label = "refitting n times")
vlines!(ax4, [lam_hat]; color = PALETTE.grey, linestyle = :dot)
axislegend(ax4; position = :lt, labelsize = 12)
ticks = [1e-5, 1e-3, 1e-1]
ax4b = Axis(fig[1, 2]; xscale = log10, xaxisposition = :top, xlabel = L"\text{degrees of freedom } \mathrm{tr}\, H_\lambda",
            xticks = (ticks, [string(round(Int, loo(t)[2])) for t in ticks]), xgridvisible = false, ygridvisible = false,
            backgroundcolor = :transparent)
hideydecorations!(ax4b); hidespines!(ax4b)
linkxaxes!(ax4, ax4b)
save_fig(fig, @__DIR__, "penalty-selection")
