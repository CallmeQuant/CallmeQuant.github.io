# Dirichlet process mixture of bivariate normals (known covariance sigma^2 I, base measure
# N(mu0, tau0^2 I)) fitted by collapsed Gibbs sampling (Neal's Algorithm 3): data and one posterior
# partition, trace of the number of clusters, and prior versus posterior of K.
# Packages: CairoMakie, Random, Statistics, StatsBase
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, StatsBase

set_blog_theme!()
rng = MersenneTwister(2000)
centers = [0.0 0.0; 3.0 0.0; 0.0 3.0; 3.5 3.5]
props = [0.35, 0.30, 0.20, 0.15]
n, sigma = 300, 0.5
labels = sample(rng, 1:4, Weights(props), n)
X = centers[labels, :] .+ sigma .* randn(rng, n, 2)
alpha, tau0 = 1.0, 3.0
mu0 = vec(mean(X; dims = 1))

"log predictive density of x for clusters with counts cnt and coordinate sums sm (K x 2)."
function log_pred(x, cnt, sm)
    s2 = 1 ./ (1 / tau0^2 .+ cnt ./ sigma^2)
    m = s2 .* (mu0' ./ tau0^2 .+ sm ./ sigma^2)
    v = sigma^2 .+ s2
    return -log.(2pi .* v) .- vec(sum((x' .- m) .^ 2; dims = 2)) ./ (2 .* v)
end

function run_gibbs(rng, X, sweeps)
    n = size(X, 1)
    z = ones(Int, n); cnt = [Float64(n)]; sm = sum(X; dims = 1)
    trace = zeros(Int, sweeps)
    for it in 1:sweeps
        for i in randperm(rng, n)
            k = z[i]; cnt[k] -= 1; sm[k, :] .-= X[i, :]
            if cnt[k] == 0                     # drop the emptied cluster
                deleteat!(cnt, k); sm = sm[setdiff(1:size(sm, 1), k), :]; z[z .> k] .-= 1
            end
            lp = vcat(log.(cnt) .+ log_pred(X[i, :], cnt, sm),
                      log(alpha) + log_pred(X[i, :], [0.0], zeros(1, 2))[1])
            p = exp.(lp .- maximum(lp))
            k = sample(rng, 1:length(p), Weights(p))
            if k > length(cnt)                 # open a new cluster
                push!(cnt, 0.0); sm = vcat(sm, zeros(1, 2))
            end
            z[i] = k; cnt[k] += 1; sm[k, :] .+= X[i, :]
        end
        trace[it] = length(cnt)
    end
    return z, trace
end

sweeps, burn = 400, 100
z, trace = run_gibbs(rng, X, sweeps)

rank_by_size(lab) = (r = invperm(sortperm(counts(lab, 1:maximum(lab)); rev = true)); r[lab])
cols(lab) = [CYCLE[mod1(r, length(CYCLE))] for r in rank_by_size(lab)]

fig = Figure(size = (780, 380))
ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"x_2", title = "Data, coloured by true component", aspect = DataAspect())
scatter!(ax1, X[:, 1], X[:, 2]; color = cols(labels), markersize = 6)
ax2 = Axis(fig[1, 2]; xlabel = L"x_1", title = "Partition at sweep $sweeps", aspect = DataAspect())
scatter!(ax2, X[:, 1], X[:, 2]; color = cols(z), markersize = 6)
linkaxes!(ax1, ax2); hideydecorations!(ax2; grid = false)
save_fig(fig, @__DIR__, "dpmm-partition")

pk = [1.0]
for i in 1:n
    q = alpha / (alpha + i - 1)
    global pk = vcat(pk .* (1 - q), 0.0) .+ vcat(0.0, pk .* q)
end
ks = 1:15
post = [mean(trace[burn+1:end] .== k) for k in ks]

fig2 = Figure(size = (780, 300))
ax3 = Axis(fig2[1, 1]; xlabel = "sweep", ylabel = L"\text{number of clusters } K", title = "Trace of the number of clusters")
stairs!(ax3, 1:sweeps, trace; step = :center, color = PALETTE.blue, linewidth = 1)
vlines!(ax3, [burn]; color = PALETTE.grey, linestyle = :dash, linewidth = 1)
text!(ax3, burn + 6, 10.2; text = "burn-in ends", fontsize = 12, color = PALETTE.grey)
xlims!(ax3, 0, sweeps); ylims!(ax3, 0, 11)
ax4 = Axis(fig2[1, 2]; xlabel = L"K", ylabel = "probability", title = L"\text{Prior and posterior of } K,\ \alpha = 1",
           xticks = [1, 4, 7, 10, 13], xgridvisible = false)
barplot!(ax4, ks .- 0.2, pk[ks .+ 1]; width = 0.4, color = PALETTE.grey, label = "prior")
barplot!(ax4, ks .+ 0.2, post; width = 0.4, color = PALETTE.blue, label = "posterior")
xlims!(ax4, 0.3, 15.7)
axislegend(ax4; position = :rt)
colsize!(fig2.layout, 2, Relative(0.42))
save_fig(fig2, @__DIR__, "dpmm-clusters")
