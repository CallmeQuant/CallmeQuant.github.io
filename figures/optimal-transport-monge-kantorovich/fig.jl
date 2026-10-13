# Optimal transport: (a) optimal assignment between two 7-point clouds (squared Euclidean cost,
# all permutations enumerated); (b) the monotone (quantile) coupling between two densities on
# the line; entropic transport plans (log-domain Sinkhorn) for three regularization levels.
# Packages: CairoMakie, Distributions, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random

set_blog_theme!()
rng = MersenneTwister(1781)

# ---------------------------------------------------------------- (a) assignment
n = 7
X = hcat(0.55 .* randn(rng, n) .- 1.3, 0.55 .* randn(rng, n))
ang = 2pi .* rand(rng, n)
Y = hcat(1.1 .* cos.(ang) .+ 1.3, 1.1 .* sin.(ang)) .+ 0.08 .* randn(rng, n, 2)
C = [sum((X[i, :] .- Y[j, :]) .^ 2) for i in 1:n, j in 1:n]

function allperms(v)
    length(v) == 1 && return [v]
    out = Vector{Vector{Int}}()
    for i in eachindex(v), p in allperms(deleteat!(copy(v), i))
        push!(out, vcat(v[i], p))
    end
    return out
end
P = allperms(collect(1:n))
best = P[argmin([sum(C[i, p[i]] for i in 1:n) for p in P])]

fig = Figure(size = (840, 350))
a0 = Axis(fig[1, 1]; aspect = DataAspect(), title = L"\text{(a) optimal assignment, } c(x,y) = \Vert x - y\Vert^2",
          xticks = [-2, 0, 2], yticks = [-1, 0, 1, 2])
for i in 1:n
    lines!(a0, [X[i, 1], Y[best[i], 1]], [X[i, 2], Y[best[i], 2]]; color = PALETTE.grey, linewidth = 1.3)
end
scatter!(a0, X[:, 1], X[:, 2]; color = PALETTE.blue, markersize = 10, label = L"x_i \text{ (mass } 1/7 \text{ each)}")
scatter!(a0, Y[:, 1], Y[:, 2]; color = PALETTE.red, marker = :rect, markersize = 9, label = L"y_j \text{ (mass } 1/7 \text{ each)}")
xlims!(a0, -2.6, 2.8); ylims!(a0, -1.7, 2.3)
axislegend(a0; position = :lt)

# ---------------------------------------------------------------- (b) one dimension
f_mu(x) = 0.55 * pdf(Normal(-1.6, 0.55), x) + 0.45 * pdf(Normal(1.0, 0.7), x)
f_nu(x) = pdf(Normal(0.6, 1.0), x)
grid = collect(range(-5, 5, length = 4001))
Fc = cumsum(f_mu.(grid)); Fc ./= Fc[end]
Gc = cumsum(f_nu.(grid)); Gc ./= Gc[end]
"Piecewise-linear interpolation of (xp, yp) at t, with xp nondecreasing; constant outside."
function interp(t, xp, yp)
    t <= xp[1] && return yp[1]
    t >= xp[end] && return yp[end]
    k = searchsortedlast(xp, t)
    xp[k+1] == xp[k] && return yp[k]
    return yp[k] + (yp[k+1] - yp[k]) * (t - xp[k]) / (xp[k+1] - xp[k])
end
Finv(t) = interp(t, Fc, grid)
Ginv(t) = interp(t, Gc, grid)

a1 = Axis(fig[1, 2]; xlabel = L"x", title = L"\text{(b) quantile coupling } F^{-1}(t) \mapsto G^{-1}(t)",
          xgridvisible = false, ygridvisible = false, leftspinevisible = false,
          yticksvisible = false, yticklabelsvisible = false)
xs = range(-4, 4, length = 600); top = 1.0; s = 0.9
band!(a1, xs, fill(top, length(xs)), top .+ s .* f_mu.(xs); color = (PALETTE.blue, 0.25))
lines!(a1, xs, top .+ s .* f_mu.(xs); color = PALETTE.blue, label = L"\mu")
band!(a1, xs, -s .* f_nu.(xs), zeros(length(xs)); color = (PALETTE.red, 0.25))
lines!(a1, xs, -s .* f_nu.(xs); color = PALETTE.red, label = L"\nu")
hlines!(a1, [0.0, top]; color = PALETTE.ink, linewidth = 0.8)
for t in ((0:14) .+ 0.5) ./ 15
    x0, x1 = Finv(t), Ginv(t)
    lines!(a1, [x0, x1], [top - 0.02, 0.05]; color = PALETTE.ink, linewidth = 0.8)
    scatter!(a1, [x1], [0.05]; marker = :dtriangle, markersize = 7, color = PALETTE.ink)
end
xlims!(a1, -4, 4); ylims!(a1, -0.5, 1.5)
axislegend(a1; position = :rt)
colsize!(fig.layout, 2, Relative(0.53))
save_fig(fig, @__DIR__, "couplings")

# ---------------------------------------------------------------- entropic plans
m = 120
z = collect(range(-4, 4, length = m))
a = f_mu.(z); a ./= sum(a)
b = f_nu.(z); b ./= sum(b)
Cz = [(x - y)^2 for x in z, y in z]
lse(v) = (mx = maximum(v); mx + log(sum(exp.(v .- mx))))
function sinkhorn_log(a, b, C, eps, iters)
    f = zeros(length(a)); g = zeros(length(b))
    for _ in 1:iters
        f = [eps * log(a[i]) - eps * lse((g .- C[i, :]) ./ eps) for i in eachindex(a)]
        g = [eps * log(b[j]) - eps * lse((f .- C[:, j]) ./ eps) for j in eachindex(b)]
    end
    return exp.((f .+ g' .- C) ./ eps)
end
Tmap = [Ginv(interp(x, grid, Fc)) for x in z]
cmap = cgrad([colorant"white", PALETTE.blue, PALETTE.ink])
fig2 = Figure(size = (840, 320))
for (k, eps) in enumerate([1.0, 0.1, 0.01])
    Pm = sinkhorn_log(a, b, Cz, eps, 3000)
    ax = Axis(fig2[1, k]; aspect = DataAspect(), title = L"\varepsilon = %$(eps)",
              xlabel = L"x \text{ (source } \mu)", ylabel = k == 1 ? L"y \text{ (target } \nu)" : "",
              xgridvisible = false, ygridvisible = false)
    heatmap!(ax, z, z, Pm; colormap = cmap, colorrange = (0, maximum(Pm)))
    lines!(ax, z, Tmap; color = PALETTE.red, linestyle = :dash, linewidth = 1.2)
    xlims!(ax, -4, 4); ylims!(ax, -4, 4)
    k > 1 && hideydecorations!(ax; ticks = false)
end
save_fig(fig2, @__DIR__, "entropic")
