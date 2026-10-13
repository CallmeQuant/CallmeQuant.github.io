# One posterior, four computations: a two-parameter logistic regression (n = 25) computed exactly on
# a grid, by the Laplace approximation, by random-walk Metropolis and by mean-field Gaussian
# variational inference.
# Packages: CairoMakie, Distributions, LinearAlgebra, Optim, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Optim, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1986)
n, prior_sd = 25, 2.5
x = 3 .* rand(rng, n)
logistic(t) = 1 / (1 + exp(-t))
log_logistic(t) = -log1p(exp(-t))
y = Float64.(rand(rng, n) .< logistic.(-2.0 .+ 1.5 .* x))
loglik(a, b) = sum(y .* log_logistic.(a .+ b .* x) .+ (1 .- y) .* log_logistic.(-(a .+ b .* x)))
logpost(a, b) = loglik(a, b) - (a^2 + b^2) / (2 * prior_sd^2)
lp(t) = logpost(t[1], t[2])

mle = Optim.minimizer(optimize(t -> -loglik(t[1], t[2]), [0.0, 0.0], BFGS()))
mode = Optim.minimizer(optimize(t -> -lp(t), [0.0, 0.0], BFGS()))
function hessian(f, t; h = 1e-4)
    H = zeros(2, 2); E = Matrix(1.0I, 2, 2) .* h
    for i in 1:2, j in 1:2
        ei, ej = E[:, i], E[:, j]
        H[i, j] = (f(t + ei + ej) - f(t + ei - ej) - f(t - ei + ej) + f(t - ei - ej)) / (4h^2)
    end
    return H
end
lap_cov = inv(-hessian(lp, mode))

A = range(-5.5, 3.0; length = 501); B = range(-1.5, 3.5; length = 501)
dA, dB = step(A), step(B)
LP = [logpost(a, b) for a in A, b in B]
P = exp.(LP .- maximum(LP)); P ./= sum(P) * dA * dB
post_mean = [sum(a * P[i, j] for (i, a) in enumerate(A), j in eachindex(B)),
             sum(b * P[i, j] for i in eachindex(A), (j, b) in enumerate(B))] .* (dA * dB)

iters, burn = 20000, 2000
Lc = cholesky(Symmetric(2.38^2 / 2 .* lap_cov)).L
cur = copy(mode); cur_lp = lp(cur); chain = zeros(iters, 2)
for t in 1:iters
    prop = cur .+ Lc * randn(rng, 2); prop_lp = lp(prop)
    if log(rand(rng)) < prop_lp - cur_lp
        global cur = prop
        global cur_lp = prop_lp
    end
    chain[t, :] .= cur
end
chain = chain[burn+1:end, :]

# probabilists' Gauss-Hermite rule (Golub-Welsch)
function gauss_hermite(m)
    e = eigen(SymTridiagonal(zeros(m), sqrt.(1.0:m-1)))
    return e.values, e.vectors[1, :] .^ 2
end
ghx, ghw = gauss_hermite(30)
function neg_elbo(par)
    a = par[1] .+ exp(par[3]) .* ghx; b = par[2] .+ exp(par[4]) .* ghx
    return -(sum(ghw[i] * ghw[j] * logpost(a[i], b[j]) for i in eachindex(a), j in eachindex(b)) + par[3] + par[4])
end
vi = Optim.minimizer(optimize(neg_elbo, [mode; 0.0; 0.0], NelderMead(), Optim.Options(iterations = 5000, g_tol = 1e-12)))
vi_m, vi_s = vi[1:2], exp.(vi[3:4])

gauss_density(m, C) = [pdf(MvNormal(m, Symmetric(C)), [a, b]) for a in A, b in B]
function levels_of(D)
    s = sort(vec(D); rev = true); c = cumsum(s) .* (dA * dB)
    return sort([s[findfirst(>=(q), c)] for q in (0.99, 0.9, 0.5)])
end

fig = Figure(size = (720, 660))
titles = ["Exact (grid)", "Laplace approximation", "Random-walk Metropolis", "Mean-field variational"]
Qs = [nothing, gauss_density(mode, lap_cov), nothing, gauss_density(vi_m, Matrix(Diagonal(vi_s .^ 2)))]
axs = Axis[]
for k in 1:4
    r, c = fldmod1(k, 2)
    ax = Axis(fig[r, c]; title = titles[k], xlabel = r == 2 ? L"\text{intercept } \alpha" : "",
              ylabel = c == 1 ? L"\text{slope } \beta" : "")
    contour!(ax, A, B, P; levels = levels_of(P), color = k == 1 ? PALETTE.ink : PALETTE.grey, linewidth = 1.2)
    if Qs[k] !== nothing
        contour!(ax, A, B, Qs[k]; levels = levels_of(Qs[k]), color = CYCLE[k], linestyle = :dash, linewidth = 1.6)
    end
    if k == 3
        idx = randperm(rng, size(chain, 1))[1:1500]
        scatter!(ax, chain[idx, 1], chain[idx, 2]; color = (PALETTE.green, 0.5), markersize = 3)
    end
    if k == 1
        scatter!(ax, [mle[1]], [mle[2]]; marker = :rect, color = PALETTE.red, markersize = 9, label = "MLE")
        scatter!(ax, [mode[1]], [mode[2]]; color = PALETTE.blue, markersize = 9, label = "MAP")
        scatter!(ax, [post_mean[1]], [post_mean[2]]; marker = :utriangle, color = PALETTE.gold, markersize = 10,
                 label = "posterior mean")
        axislegend(ax; position = :lb, labelsize = 12)
    end
    xlims!(ax, -4.2, 1.8); ylims!(ax, -1.0, 2.6)
    push!(axs, ax)
end
linkaxes!(axs...)
save_fig(fig, @__DIR__, "four-posteriors")

pb = vec(sum(P; dims = 1)) .* dA
fig2 = Figure(size = (640, 320))
ax = Axis(fig2[1, 1]; xlabel = L"\text{slope } \beta", ylabel = "marginal posterior density")
hist!(ax, chain[:, 2]; bins = range(-0.6, 2.6; length = 71), normalization = :pdf, color = PALETTE.light,
      strokecolor = :white, strokewidth = 0.5, label = "Metropolis samples")
lines!(ax, B, pb; color = PALETTE.ink, linewidth = 2, label = "exact")
lines!(ax, B, pdf.(Normal(mode[2], sqrt(lap_cov[2, 2])), B); color = PALETTE.red, linestyle = :dash, linewidth = 1.8,
       label = "Laplace")
lines!(ax, B, pdf.(Normal(vi_m[2], vi_s[2]), B); color = PALETTE.violet, linestyle = :dot, linewidth = 1.8,
       label = "mean-field VI")
xlims!(ax, -0.6, 2.6); ylims!(ax, 0, 1.15 * max(maximum(pb), pdf(Normal(0, vi_s[2]), 0)))
axislegend(ax; position = :rt)
save_fig(fig2, @__DIR__, "slope-marginal")
