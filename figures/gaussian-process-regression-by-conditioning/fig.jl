# Gaussian process regression: prior draws for three kernels, the posterior given noisy data, and the
# log marginal likelihood as a function of the length-scale.
# Packages: CairoMakie, Random, LinearAlgebra, Optim
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, LinearAlgebra, Optim

set_blog_theme!()
rng = MersenneTwister(1996)
k_se(a, b, ell, sf = 1.0) = sf^2 .* exp.(-0.5 .* (a .- b') .^ 2 ./ ell^2)
k_exp(a, b, ell, sf = 1.0) = sf^2 .* exp.(-abs.(a .- b') ./ ell)
xs = collect(range(-3, 3; length = 400))

# ---------- Figure 1 ----------
kernels = [(L"\text{squared exponential, } \ell = 1", (a, b) -> k_se(a, b, 1.0)),
           (L"\text{squared exponential, } \ell = 0.25", (a, b) -> k_se(a, b, 0.25)),
           (L"\text{exponential, } \ell = 1", (a, b) -> k_exp(a, b, 1.0))]
fig = Figure(size = (820, 280))
axs = Axis[]
for (j, (ttl, kf)) in enumerate(kernels)
    ax = Axis(fig[1, j]; title = ttl, xlabel = L"x", ylabel = j == 1 ? L"f(x)" : "", titlesize = 13)
    L = cholesky(Symmetric(kf(xs, xs) + 1e-9I)).L
    band!(ax, [-3.0, 3.0], [-2.0, -2.0], [2.0, 2.0]; color = PALETTE.light)
    for d in 1:4
        lines!(ax, xs, L * randn(rng, length(xs)); color = CYCLE[d], linewidth = 1.3)
    end
    xlims!(ax, -3, 3); ylims!(ax, -3.2, 3.2)
    push!(axs, ax)
end
linkyaxes!(axs...)
hideydecorations!(axs[2]; grid = false); hideydecorations!(axs[3]; grid = false)
save_fig(fig, @__DIR__, "prior-draws")

# ---------- data and marginal likelihood ----------
f_true(x) = sin(1.5x) + 0.3x
n = 15
X = sort(6 .* rand(rng, n) .- 3)
y = f_true.(X) .+ 0.25 .* randn(rng, n)
function nlml_parts(ell, sf, sn)
    C = cholesky(Symmetric(k_se(X, X, ell, sf) + sn^2 * I))
    return 0.5 * dot(y, C \ y), sum(log.(diag(C.L))), 0.5 * n * log(2π)
end
nlml(p) = sum(nlml_parts(exp.(p)...))
starts = [log.([0.5, 1.0, 0.3]), log.([2.0, 1.0, 0.1]), log.([0.2, 0.5, 0.5])]
res = [optimize(nlml, p0, LBFGS()) for p0 in starts]
best = res[argmin(Optim.minimum.(res))]
ell_hat, sf_hat, sn_hat = exp.(Optim.minimizer(best))
function posterior(xq, ell, sf, sn)
    C = cholesky(Symmetric(k_se(X, X, ell, sf) + sn^2 * I))
    Ks = k_se(xq, X, ell, sf)
    return Ks * (C \ y), k_se(xq, xq, ell, sf) - Ks * (C \ Ks')
end

# ---------- Figure 2 ----------
mn, cv = posterior(xs, ell_hat, sf_hat, sn_hat)
sd = sqrt.(max.(diag(cv), 0))
fig = Figure(size = (660, 330))
ax = Axis(fig[1, 1]; xlabel = L"x", ylabel = L"f(x)")
band!(ax, xs, mn .- 1.96 .* sd, mn .+ 1.96 .* sd; color = (PALETTE.blue, 0.15), label = "95% pointwise band for f")
Lp = cholesky(Symmetric(cv + 1e-8I)).L
for j in 1:3
    lines!(ax, xs, mn .+ Lp * randn(rng, length(xs)); color = (PALETTE.blue, 0.6), linewidth = 0.8,
           label = j == 1 ? "posterior draws" : nothing)
end
lines!(ax, xs, mn; color = PALETTE.blue, linewidth = 2.4, label = "posterior mean")
lines!(ax, xs, f_true.(xs); color = PALETTE.ink, linestyle = :dash, linewidth = 1.4, label = L"\text{true } f")
scatter!(ax, X, y; color = PALETTE.red, markersize = 9, label = "observations")
xlims!(ax, -3, 3); ylims!(ax, -2.6, 2.6)
axislegend(ax; position = :lt, nbanks = 2, labelsize = 12)
save_fig(fig, @__DIR__, "posterior")

# ---------- Figure 3 ----------
ells = exp.(range(log(0.05), log(10); length = 300))
parts = [nlml_parts(l, sf_hat, sn_hat) for l in ells]
fig = Figure(size = (800, 330))
ax1 = Axis(fig[1, 1]; xscale = log10, xlabel = L"\text{length-scale } \ell", ylabel = "nats",
           title = L"\text{Marginal likelihood } (\sigma_f,\ \sigma \text{ fixed})")
lines!(ax1, ells, [-sum(p) for p in parts]; color = PALETTE.ink, label = L"\log p(y \mid \ell)")
lines!(ax1, ells, [-p[1] for p in parts]; color = PALETTE.red, linestyle = :dash, label = L"-\frac{1}{2} y^\top K_y^{-1} y \text{ (fit)}")
lines!(ax1, ells, [-p[2] for p in parts]; color = PALETTE.blue, linestyle = :dash, label = L"-\frac{1}{2}\log\det K_y \text{ (complexity)}")
vlines!(ax1, [ell_hat]; color = PALETTE.grey, linestyle = :dot)
xlims!(ax1, 0.05, 10); ylims!(ax1, -40, 25)
axislegend(ax1; position = :lb, labelsize = 11)
ax2 = Axis(fig[1, 2]; xlabel = L"x", title = "Posterior means")
for (k, l) in enumerate([0.08, ell_hat, 6.0])
    lines!(ax2, xs, posterior(xs, l, sf_hat, sn_hat)[1]; color = CYCLE[[4, 1, 3][k]], linewidth = 1.8,
           label = L"\ell = %$(round(l; digits = 2))")
end
scatter!(ax2, X, y; color = PALETTE.red, markersize = 8)
xlims!(ax2, -3, 3); ylims!(ax2, -2.6, 2.6)
axislegend(ax2; position = :lt, labelsize = 12)
save_fig(fig, @__DIR__, "marginal-likelihood")
