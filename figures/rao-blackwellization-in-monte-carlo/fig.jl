# Rao-Blackwellization in a two-block Gibbs sampler for Student t_3 (data augmentation).
# Y | X ~ Gamma((nu+1)/2, rate (nu+X^2)/2), X | Y ~ N(0, 1/Y).
# Packages: CairoMakie, Distributions, Random, Statistics, KernelDensity, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, KernelDensity, Printf

set_blog_theme!()
rng = MersenneTwister(1990)
const NU, N, A = 3.0, 500, 3.0

function gibbs(n_iter, n_chains)
    X = zeros(n_chains, n_iter); Y = zeros(n_chains, n_iter)
    x = rand(rng, TDist(NU), n_chains)
    for t in 1:n_iter
        y = [rand(rng, Gamma((NU + 1) / 2, 2 / (NU + xi^2))) for xi in x]   # shape, scale
        x = randn(rng, n_chains) ./ sqrt.(y)
        Y[:, t] = y; X[:, t] = x
    end
    return X, Y
end

fig = Figure(size = (790, 350))
X, Y = gibbs(N, 1)
x, y = X[1, :], Y[1, :]
grid = range(-8, 8; length = 801)
kd = kde(x)
kgrid = pdf(kd, grid)
rb = [mean(pdf.(Normal.(0, 1 ./ sqrt.(y)), g)) for g in grid]
ax1 = Axis(fig[1, 1]; title = L"\text{One chain, } N = 500", xlabel = L"x", ylabel = "density (log scale)", yscale = log10)
lines!(ax1, grid, pdf.(TDist(NU), grid); color = PALETTE.ink, linestyle = :dash, label = L"\text{target } t_3 \text{ density}")
keep = kgrid .> 1e-5
lines!(ax1, grid[keep], kgrid[keep]; color = PALETTE.grey, label = L"\text{kernel estimate from } X_t")
lines!(ax1, grid, rb; color = PALETTE.blue, label = L"\frac{1}{N}\sum_t \pi(x \mid Y_t)")
xlims!(ax1, -8, 8); ylims!(ax1, 1e-4, 0.8)
axislegend(ax1; position = :cb, labelsize = 12)

X, Y = gibbs(N, 4000)
crude = vec(mean(X .> A; dims = 2))
rbp = vec(mean(ccdf.(Normal(), A .* sqrt.(Y)); dims = 2))
truth = ccdf(TDist(NU), A)
edges = ((0:36) .- 0.5) ./ N
ax2 = Axis(fig[1, 2]; title = L"4000 \text{ chains, } N = 500 \text{ each}",
           xlabel = @sprintf("estimate of P(X > 3) = %.4f", truth), ylabel = "number of chains")
hist!(ax2, crude; bins = edges, color = (PALETTE.grey, 0.55), label = @sprintf("crude: sd %.4f", std(crude)))
hist!(ax2, rbp; bins = edges, color = (PALETTE.blue, 0.75), label = @sprintf("Rao–Blackwell: sd %.4f", std(rbp)))
vlines!(ax2, [truth]; color = PALETTE.ink, linestyle = :dash, linewidth = 1.2)
axislegend(ax2; position = :rt, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.54))
save_fig(fig, @__DIR__, "rb-gibbs")
