# AdaBoost with decision stumps on a disc-shaped class: decision region, error curves
# with the exponential-loss bound, and normalized margin distributions.
# Packages: CairoMakie, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics, Printf

set_blog_theme!()
rng = MersenneTwister(2718)
const R2 = 0.55
N_TRAIN, N_TEST, M = 200, 5000, 400
SNAPSHOTS = [5, 50, 400]

function sample_xy(n)
    x = 2 .* rand(rng, n, 2) .- 1
    y = [x[i, 1]^2 + x[i, 2]^2 < R2 ? 1.0 : -1.0 for i in 1:n]
    return x, y
end

"Weighted decision stump h(x) = s * sign(x_j - t); returns (j, t, s, weighted error)."
function fit_stump(x, y, w)
    best = (1, 0.0, 1.0, Inf)
    n = length(y)
    for j in 1:2
        o = sortperm(x[:, j]); xs = x[o, j]; ys = y[o]; ws = w[o]
        left_pos = [0.0; cumsum(ws .* (ys .== 1))]
        right_neg = reverse([0.0; cumsum(reverse(ws .* (ys .== -1)))])
        err = left_pos .+ right_neg
        cuts = [xs[1] - 1e-9; (xs[1:end-1] .+ xs[2:end]) ./ 2; xs[end] + 1e-9]
        for (s, e) in ((1.0, err), (-1.0, 1 .- err))
            k = argmin(e)
            if e[k] < best[4]
                best = (j, cuts[k], s, e[k])
            end
        end
    end
    return best
end

stump_predict(x, st) = st[3] .* ifelse.(x[:, st[1]] .> st[2], 1.0, -1.0)

function adaboost(x, y, rounds)
    w = fill(1 / length(y), length(y))
    stumps = Tuple{Int,Float64,Float64}[]; alphas = Float64[]; Z = Float64[]
    for _ in 1:rounds
        j, t, s, eps = fit_stump(x, y, w)
        a = 0.5 * log((1 - eps) / eps)
        w = w .* exp.(-a .* y .* stump_predict(x, (j, t, s)))
        push!(Z, sum(w)); w ./= sum(w)
        push!(stumps, (j, t, s)); push!(alphas, a)
    end
    return stumps, alphas, Z
end

"Matrix of F_m(x): rows = observations, columns = rounds."
function score(x, stumps, alphas)
    H = reduce(hcat, [stump_predict(x, st) for st in stumps])
    return cumsum(H .* alphas', dims = 2)
end

xtr, ytr = sample_xy(N_TRAIN)
xte, yte = sample_xy(N_TEST)
stumps, alphas, Z = adaboost(xtr, ytr, M)
Ftr = score(xtr, stumps, alphas); Fte = score(xte, stumps, alphas)
train_err = vec(mean(sign.(Ftr) .!= ytr, dims = 1))
test_err = vec(mean(sign.(Fte) .!= yte, dims = 1))
ms = 1:M

# ---- Figure 1 ----
fig = Figure(size = (780, 360))
ax0 = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = L"x_1", ylabel = L"x_2",
           title = L"\mathrm{sign}\, F_M \text{ after } M = 50 \text{ stumps}",
           xgridvisible = false, ygridvisible = false, xticks = [-1, 0, 1], yticks = [-1, 0, 1])
g = range(-1, 1, length = 301)
G = hcat(repeat(collect(g), outer = length(g)), repeat(collect(g), inner = length(g)))
Fg = reshape(score(G, stumps[1:SNAPSHOTS[2]], alphas[1:SNAPSHOTS[2]])[:, end], length(g), length(g))
heatmap!(ax0, g, g, sign.(Fg); colormap = [colorant"#f3e3dd", colorant"#dfe5f2"], colorrange = (-1, 1))
contour!(ax0, g, g, Fg; levels = [0.0], color = PALETTE.ink, linewidth = 1.2)
th = range(0, 2pi, length = 400)
lines!(ax0, sqrt(R2) .* cos.(th), sqrt(R2) .* sin.(th); color = PALETTE.grey, linestyle = :dash, linewidth = 1.2)
pos = ytr .== 1
scatter!(ax0, xtr[pos, 1], xtr[pos, 2]; color = PALETTE.blue, markersize = 6)
scatter!(ax0, xtr[.!pos, 1], xtr[.!pos, 2]; color = PALETTE.red, marker = :xcross, markersize = 7)
xlims!(ax0, -1, 1); ylims!(ax0, -1, 1)

ax1 = Axis(fig[1, 2]; xscale = log10, xlabel = L"\text{number of rounds } M",
           title = "error rates and the exponential-loss bound")
lines!(ax1, ms, cumprod(Z); color = PALETTE.gold, label = L"\text{bound } Z_1 Z_2 \cdots Z_M")
lines!(ax1, ms, train_err; color = PALETTE.blue, label = "training error")
lines!(ax1, ms, test_err; color = PALETTE.red, label = "test error")
xlims!(ax1, 1, M); ylims!(ax1, 0, 1.02)
axislegend(ax1; position = :rt)
colsize!(fig.layout, 2, Relative(0.57))
save_fig(fig, @__DIR__, "rounds")

# ---- Figure 2 ----
fig2 = Figure(size = (600, 330))
ax = Axis(fig2[1, 1]; xlabel = L"\text{normalized margin } y_i f_M(x_i)",
          ylabel = "fraction of training points")
for (Mm, col) in zip(SNAPSHOTS, [PALETTE.grey, PALETTE.green, PALETTE.violet])
    marg = sort(ytr .* Ftr[:, Mm] ./ sum(alphas[1:Mm]))
    xs = [-1.0; marg; 1.0]
    ys = [0.0; collect(1:N_TRAIN) ./ N_TRAIN; 1.0]
    stairs!(ax, xs, ys; step = :post, color = col, linewidth = 2,
            label = @sprintf("M = %d (train err %.3f)", Mm, train_err[Mm]))
end
vlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.8)
xlims!(ax, -0.3, 0.6); ylims!(ax, 0, 1.02)
axislegend(ax; position = :rb)
save_fig(fig2, @__DIR__, "margins")
