# Probabilistic PCA: sample versus fitted eigenvalues, and EM convergence to the closed-form MLE.
# Packages: CairoMakie, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Printf, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(1999)
n, p, q, sigma2 = 200, 20, 3, 1.0
W_true = randn(rng, p, q) * Diagonal([3.0, 2.0, 1.4])
X = randn(rng, n, q) * W_true' + sqrt(sigma2) * randn(rng, n, p)
Xc = X .- mean(X; dims = 1)
S = Symmetric(Xc' * Xc / n)
E = eigen(S); lam = reverse(E.values); U = E.vectors[:, end:-1:1]
function loglik(W, s2)
    C = W * W' + s2 * I
    return -0.5 * n * (p * log(2pi) + logdet(C) + tr(C \ S))
end
s2_ml = mean(lam[q+1:end])
W_ml = U[:, 1:q] * Diagonal(sqrt.(lam[1:q] .- s2_ml))
ll_ml = loglik(W_ml, s2_ml)

fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = L"\text{index } j", ylabel = "eigenvalue (log scale)",
           xticks = [1, 5, 10, 15, 20], title = L"\text{(a) } p = 20,\ q = 3\text{: } \hat\sigma^2 = %$(round(s2_ml; digits = 2))")
hlines!(ax1, [sigma2]; color = PALETTE.red, linestyle = :dot, linewidth = 1.3, label = L"\text{true } \sigma^2 = 1")
stairs!(ax1, 1:p, vcat(lam[1:q], fill(s2_ml, p - q)); step = :center, color = PALETTE.blue, linewidth = 2,
        label = L"\text{eigenvalues of } \hat C")
scatter!(ax1, 1:p, lam; color = PALETTE.ink, markersize = 8, label = L"\text{eigenvalues } \lambda_j \text{ of } S")
axislegend(ax1; position = :rt, labelsize = 11)

ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = L"\text{EM iteration } t \text{ (log scale)}",
           title = "(b) EM from four random starts")
T = 3000
P_true = U[:, 1:q] * U[:, 1:q]'
for i in 1:4
    W = randn(rng, p, q); s2 = 5.0
    gaps = Float64[]; sub = Float64[]
    for it in 1:T
        push!(gaps, ll_ml - loglik(W, s2))
        push!(sub, opnorm(W * pinv(W) - P_true))
        M = W' * W + s2 * I; Minv = inv(M); SW = S * W
        W_new = SW / (s2 * I + Minv * W' * SW)
        s2 = tr(S - SW * Minv * W_new') / p
        W = W_new
    end
    g_end = something(findfirst(<(1e-10), gaps), T)
    s_end = something(findfirst(<(1e-13), sub), T)
    lines!(ax2, 1:g_end, gaps[1:g_end]; color = CYCLE[i], linewidth = 1.6,
           label = i == 1 ? "log-likelihood gap" : nothing)
    lines!(ax2, 1:s_end, sub[1:s_end]; color = CYCLE[i], linewidth = 1.2, linestyle = :dash,
           label = i == 1 ? "distance to principal subspace" : nothing)
end
ylims!(ax2, 1e-14, 1e5); xlims!(ax2, 1, T)
axislegend(ax2; position = (0.6, 0.02), labelsize = 11)
save_fig(fig, @__DIR__, "ppca-fit")
