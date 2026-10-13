# EM for a two-component Poisson mixture on simulated counts (note: the EM algorithm).
# Packages: CairoMakie, Distributions, Random, StatsFuns
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, StatsFuns

set_blog_theme!()
rng = MersenneTwister(2023)
n, pi_true, lam_true = 500, [0.6, 0.4], [2.0, 9.0]
z = rand(rng, Categorical(pi_true), n)
x = [rand(rng, Poisson(lam_true[zi])) for zi in z]

logw(pi, lam) = [log(pi[k]) + logpdf(Poisson(lam[k]), xi) for xi in x, k in 1:2]
loglik(pi, lam) = sum(logsumexp(row) for row in eachrow(logw(pi, lam)))
function em(pi, lam, iters)
    pi, lam = float.(pi), float.(lam)
    ll = [loglik(pi, lam)]
    for _ in 1:iters
        W = logw(pi, lam)                                     # E-step
        r = exp.(W .- [logsumexp(row) for row in eachrow(W)])
        Nk = vec(sum(r; dims = 1))                            # M-step
        pi = Nk ./ n
        lam = vec(sum(r .* x; dims = 1)) ./ Nk
        push!(ll, loglik(pi, lam))
    end
    return pi, lam, ll
end

pi_hat, lam_hat, _ = em([0.5, 0.5], [1.0, 3.0], 2000)
ll_hat = loglik(pi_hat, lam_hat)

fig = Figure(size = (820, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{count } x", ylabel = "relative frequency",
           title = L"\text{(a) } n = 500 \text{ counts and the EM fit}")
k = 0:(maximum(x) + 1)
freq = [count(==(kk), x) / n for kk in k]
barplot!(ax1, collect(k), freq; color = PALETTE.light, strokecolor = PALETTE.grey, strokewidth = 0.5,
         gap = 0.15, label = "data")
for j in 1:2
    lines!(ax1, collect(k), pi_hat[j] .* pdf.(Poisson(lam_hat[j]), k); color = CYCLE[j], linestyle = :dash,
           linewidth = 1.5, label = L"\hat\pi_{%$j}\,\mathrm{Pois}(\hat\lambda_{%$j})")
end
mix = [sum(pi_hat[j] * pdf(Poisson(lam_hat[j]), kk) for j in 1:2) for kk in k]
scatterlines!(ax1, collect(k), mix; color = PALETTE.ink, markersize = 5, label = "fitted mixture")
axislegend(ax1; position = :rt, labelsize = 11)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{iteration } t", ylabel = L"\ell(\hat\theta) - \ell(\theta_t)",
           yscale = log10, title = L"\text{(b) log-likelihood gap}")
starts = [([0.5, 0.5], [1.0, 3.0], "start A"), ([0.9, 0.1], [5.0, 6.0], "start B"),
          ([0.5, 0.5], [4.0, 4.0], L"\text{start C } (\lambda_1 = \lambda_2)")]
for (j, (p0, l0, lab)) in enumerate(starts)
    _, _, ll = em(p0, l0, 30)
    lines!(ax2, 0:30, max.(ll_hat .- ll, 1e-12); color = CYCLE[j], label = lab)
end
xlims!(ax2, 0, 30); ylims!(ax2, 1e-9, 3e3)
axislegend(ax2; position = :rc, labelsize = 11)
save_fig(fig, @__DIR__, "poisson-mixture-em")
