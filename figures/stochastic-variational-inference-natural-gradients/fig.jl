# Stochastic variational inference for a Gaussian mixture with unknown means: paths of the variational
# means, and suboptimality of the ELBO against data processed for batch coordinate ascent and SVI.
# Packages: CairoMakie, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, LinearAlgebra

set_blog_theme!()
rng = MersenneTwister(2013)
const N, K, d, SIG0 = 20_000, 4, 2, 10.0
centres = [-3.0 -1.0; 0.5 3.0; 3.0 -2.0; -0.5 0.0]
lab = rand(rng, 1:K, N)
X = centres[lab, :] .+ randn(rng, N, d)

rowlse(A) = (mx = maximum(A; dims = 2); vec(mx .+ log.(sum(exp.(A .- mx); dims = 2))))
function exp_ll(Xb, m, s2)
    sq = sum(Xb .^ 2; dims = 2) .- 2 .* Xb * m' .+ sum(m .^ 2; dims = 2)'
    return -0.5 * d * log(2π) .- 0.5 .* (sq .+ d .* s2')
end
function elbo_pp(m, s2)
    loc = sum(rowlse(log(1 / K) .+ exp_ll(X, m, s2)))
    pri = sum(-0.5 * d * log(2π * SIG0^2) .- (vec(sum(m .^ 2; dims = 2)) .+ d .* s2) ./ (2 * SIG0^2))
    ent = sum(0.5 * d .* log.(2π * ℯ .* s2))
    return (loc + pri + ent) / N
end
function local_stats(Xb, m, s2)
    lp = log(1 / K) .+ exp_ll(Xb, m, s2)
    phi = exp.(lp .- rowlse(lp))
    return phi' * Xb, vec(sum(phi; dims = 1))
end
init_m = 0.5 .* randn(rng, K, d)
init_s2 = ones(K)

# batch coordinate ascent
m, s2 = copy(init_m), copy(init_s2)
cavi = Float64[]
for it in 1:40
    global m, s2
    sx, sn = local_stats(X, m, s2)
    eta2 = -1 / (2 * SIG0^2) .- 0.5 .* sn
    s2 = -1 ./ (2 .* eta2)
    m = sx .* s2
    push!(cavi, elbo_pp(m, s2))
end
best = maximum(cavi)

function svi(B, steps; kappa = 0.7, tau = 1.0, nrec = 60, seed = 0)
    r = MersenneTwister(seed)
    m, s2 = copy(init_m), copy(init_s2)
    e1, e2 = m ./ s2, -1 ./ (2 .* s2)
    rec = Set(unique(round.(Int, exp.(range(0, log(steps); length = nrec)))))
    xs, gaps, path = Float64[], Float64[], [copy(m)]
    for t in 1:steps
        idx = rand(r, 1:N, B)
        sx, sn = local_stats(X[idx, :], m, s2)
        rho = (t - 1 + tau)^(-kappa)
        e1 = (1 - rho) .* e1 .+ rho .* (N / B) .* sx
        e2 = (1 - rho) .* e2 .+ rho .* (-1 / (2 * SIG0^2) .- 0.5 * (N / B) .* sn)
        s2 = -1 ./ (2 .* e2)
        m = e1 .* s2
        push!(path, copy(m))
        if t in rec
            push!(xs, t * B / N); push!(gaps, max(best - elbo_pp(m, s2), 1e-9))
        end
    end
    return xs, gaps, path
end
runs = Dict(B => svi(B, round(Int, 12 * N / B)) for B in (10, 100))

fig = Figure(size = (800, 360))
ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"x_2", title = L"\text{Paths of the variational means } (B = 10)",
           aspect = DataAspect())
sub = randperm(rng, N)[1:2500]
scatter!(ax1, X[sub, 1], X[sub, 2]; color = (PALETTE.grey, 0.35), markersize = 2.5, strokewidth = 0)
path = runs[10][3]
for k in 1:K
    px = [p[k, 1] for p in path]; py = [p[k, 2] for p in path]
    lines!(ax1, px, py; color = CYCLE[k], linewidth = 1.2)
    scatter!(ax1, [px[1]], [py[1]]; color = :white, strokecolor = CYCLE[k], strokewidth = 1.5, markersize = 8)
    scatter!(ax1, [px[end]], [py[end]]; color = CYCLE[k], markersize = 8)
end
scatter!(ax1, centres[:, 1], centres[:, 2]; marker = :xcross, color = PALETTE.ink, markersize = 11)
xlims!(ax1, -6.5, 6.5); ylims!(ax1, -5.5, 6.5)

ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "passes through the data",
           ylabel = L"(\mathcal{L}^\star - \mathcal{L}) / N", title = "Suboptimality against data processed")
cg = best .- cavi[1:12]
keep = cg .> 1e-9
scatterlines!(ax2, collect(1.0:12.0)[keep], cg[keep]; color = PALETTE.ink, marker = :rect, markersize = 7,
              label = "batch coordinate ascent")
for (k, B) in enumerate((10, 100))
    xs, gaps, _ = runs[B]
    lines!(ax2, xs, gaps; color = CYCLE[k], linewidth = 1.6, label = "SVI, minibatch B = $B")
end
xlims!(ax2, 4e-4, 13); ylims!(ax2, 1e-7, 10)
axislegend(ax2; position = :lb)
colsize!(fig.layout, 2, Relative(0.56))
save_fig(fig, @__DIR__, "svi-mixture")
