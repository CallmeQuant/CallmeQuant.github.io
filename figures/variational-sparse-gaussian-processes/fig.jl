# Titsias's collapsed variational bound for sparse Gaussian process regression: the bound against the
# exact log marginal likelihood for nested inducing sets, and the effect of optimizing the inducing inputs.
# Packages: CairoMakie, Random, LinearAlgebra, Optim
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, LinearAlgebra, Optim

set_blog_theme!()
rng = MersenneTwister(2009)
const ELL, SF, SN = 0.6, 1.0, 0.25
kern(a, b) = SF^2 .* exp.(-0.5 .* (a .- b') .^ 2 ./ ELL^2)
f_true(x) = 0.8 * sin(2x) + 0.4 * cos(5x)
n = 250
X = sort(8 .* rand(rng, n))
y = f_true.(X) .+ SN .* randn(rng, n)

function lowrank(Z)
    m = length(Z)
    Lu = cholesky(Symmetric(kern(Z, Z) + 1e-8I)).L
    V = Lu \ kern(Z, X)                                   # Q_ff = V'V
    LB = cholesky(Symmetric(Matrix(1.0I, m, m) + V * V' ./ SN^2)).L
    c = (LB \ (V * y)) ./ SN^2
    quad = dot(y, y) / SN^2 - dot(c, c)
    logdet = 2 * sum(log.(diag(LB))) + n * log(SN^2)
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * log(2π), n * SF^2 - sum(V .^ 2)
end
bound(Z) = (r = lowrank(Z); r[1] - r[2] / (2 * SN^2))
Cx = cholesky(Symmetric(kern(X, X) + SN^2 * I))
log_ev = -0.5 * dot(y, Cx \ y) - sum(log.(diag(Cx.L))) - 0.5 * n * log(2π)

perm = randperm(rng, n)
ms = collect(1:60)
R = [lowrank(X[perm[1:m]]) for m in ms]
ll = first.(R); tr = last.(R)
Lb = ll .- tr ./ (2 * SN^2)

fig = Figure(size = (780, 320))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{number of inducing inputs } m", ylabel = "nats", title = "Bound and approximate likelihood")
hlines!(ax1, [log_ev]; color = PALETTE.ink, linestyle = :dash, label = L"\text{exact } \log p(y)")
lines!(ax1, ms, ll; color = PALETTE.gold, label = L"\log\mathcal{N}(y \mid 0, Q_{ff} + \sigma^2 I)")
lines!(ax1, ms, Lb; color = PALETTE.blue, linewidth = 2.6, label = L"\text{bound } \mathcal{L}^\ast(Z)")
xlims!(ax1, 1, 60); ylims!(ax1, -420, -30)
axislegend(ax1; position = :rb, labelsize = 12)
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{number of inducing inputs } m", title = "Gap and trace penalty")
lines!(ax2, ms, log_ev .- Lb; color = PALETTE.blue, linewidth = 2.6, label = L"\text{gap } \log p(y) - \mathcal{L}^\ast")
lines!(ax2, ms, tr ./ (2 * SN^2); color = PALETTE.red, linestyle = :dash, label = L"\mathrm{tr}(K_{ff} - Q_{ff}) / (2\sigma^2)")
xlims!(ax2, 1, 60)
axislegend(ax2; position = :rt, labelsize = 12)
save_fig(fig, @__DIR__, "bound-vs-m")

function predict(xq, Z)
    Kuu = kern(Z, Z) + 1e-8I; Kuf = kern(Z, X); Ksu = kern(xq, Z)
    Cs = cholesky(Symmetric(Kuu + Kuf * Kuf' ./ SN^2)); Cu = cholesky(Symmetric(Kuu))
    mean_ = Ksu * (Cs \ (Kuf * y)) ./ SN^2
    var_ = SF^2 .- vec(sum(Ksu .* (Cu \ Ksu')'; dims = 2)) .+ vec(sum(Ksu .* (Cs \ Ksu')'; dims = 2))
    return mean_, var_
end
m = 16
Z0 = collect(range(0.2, 2.0; length = m))
res = optimize(z -> -bound(z), fill(0.0, m), fill(8.0, m), Z0, Fminbox(LBFGS()))
Zopt = sort(Optim.minimizer(res))
xs = collect(range(-0.5, 8.5; length = 500))
fig = Figure(size = (680, 460))
axs = Axis[]
for (j, (Z, ttl)) in enumerate([(Z0, "initial inducing inputs"), (Zopt, "after maximizing the bound over Z")])
    ax = Axis(fig[j, 1]; title = "$(ttl): L* = $(round(bound(Z); digits = 1))", ylabel = L"f(x)", xlabel = j == 2 ? L"x" : "")
    mu, v = predict(xs, Z)
    sd = sqrt.(max.(v, 0))
    scatter!(ax, X, y; color = (PALETTE.grey, 0.6), markersize = 3.5)
    band!(ax, xs, mu .- 1.96 .* sd, mu .+ 1.96 .* sd; color = (PALETTE.blue, 0.18))
    lines!(ax, xs, mu; color = PALETTE.blue, linewidth = 2)
    lines!(ax, xs, f_true.(xs); color = PALETTE.ink, linestyle = :dash, linewidth = 1)
    scatter!(ax, Z, fill(-2.15, m); marker = :utriangle, color = PALETTE.red, markersize = 10)
    xlims!(ax, -0.5, 8.5); ylims!(ax, -2.3, 2.3)
    push!(axs, ax)
end
linkxaxes!(axs...)
hidexdecorations!(axs[1]; grid = false)
save_fig(fig, @__DIR__, "optimized-inducing-inputs")
