# Newton's method versus Fisher scoring for a gamma GLM with identity link.
# Left: log-likelihood contours, region where the observed information is not positive definite,
# iterates from two starting points. Right: error against iteration near the MLE.
# Packages: CairoMakie, Distributions, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, LinearAlgebra

set_blog_theme!()
rng = MersenneTwister(1972)
const n, NU = 60, 2.0
x = rand(rng, n); X = hcat(ones(n), x)
y = [rand(rng, Gamma(NU, (1 + 2xi) / NU)) for xi in x]

function loglik(b)
    m = X * b
    any(m .<= 0) && return -Inf
    return NU * sum(-y ./ m .- log.(m))
end
score(b) = (m = X * b; NU .* (X' * ((y .- m) ./ m .^ 2)))
obs_info(b) = (m = X * b; NU .* ((X .* ((2 .* y .- m) ./ m .^ 3))' * X))
exp_info(b) = (m = X * b; NU .* ((X ./ m .^ 2)' * X))

function iterate_fit(b, method, iters)
    path = [collect(float.(b))]
    for _ in 1:iters
        b = path[end]
        M = method == :newton ? obs_info(b) : exp_info(b)
        step = M \ score(b); s = 1.0
        while !isfinite(loglik(b .+ s .* step)); s /= 2; end
        push!(path, b .+ s .* step)
    end
    return reduce(hcat, path)'
end
bhat = iterate_fit([1.0, 2.0], :fisher, 60)[end, :]

fig = Figure(size = (790, 360))
ax = Axis(fig[1, 1]; title = L"\text{Gamma GLM, identity link, } n = 60", xlabel = L"\beta_0", ylabel = L"\beta_1",
          xgridvisible = false, ygridvisible = false)
g0 = range(0.02, 8; length = 250); g1 = range(-8, 8; length = 250)
dev = [isfinite(loglik([u, v])) ? loglik(bhat) - loglik([u, v]) : NaN for u in g0, v in g1]
notnd = [isfinite(loglik([u, v])) && eigmin(Symmetric(obs_info([u, v]))) <= 0 ? 1.0 : NaN for u in g0, v in g1]
heatmap!(ax, g0, g1, notnd; colormap = [(PALETTE.gold, 0.22), (PALETTE.gold, 0.22)])
contour!(ax, g0, g1, dev; levels = [2, 10, 30, 60, 100, 160, 240], color = PALETTE.grey, linewidth = 0.8)
poly!(ax, Point2f[(0, 0), (8, -8), (0, -8)]; color = PALETTE.light)
for (start, mk) in [((6.0, 6.0), :circle), ((4.0, 0.5), :rect)]
    for (method, col) in [(:newton, PALETTE.red), (:fisher, PALETTE.blue)]
        P = iterate_fit(collect(start), method, 12)
        scatterlines!(ax, P[:, 1], P[:, 2]; color = col, marker = mk, markersize = 6, linewidth = 1.5)
    end
    scatter!(ax, [start[1]], [start[2]]; color = PALETTE.ink, marker = mk, markersize = 11)
end
scatter!(ax, [bhat[1]], [bhat[2]]; color = PALETTE.ink, marker = :star5, markersize = 16)
text!(ax, 4.75, 4.3; text = "observed information\nnot positive definite", fontsize = 12, color = colorant"#8a6418", align = (:left, :top))
text!(ax, 6.9, 7.55; text = "Newton diverges", fontsize = 12, color = PALETTE.red, align = (:right, :center))
text!(ax, 5.2, -7.2; text = L"\beta_0 + \beta_1 \leq 0", fontsize = 12)
xlims!(ax, 0, 8); ylims!(ax, -8, 8)
axislegend(ax, [LineElement(color = PALETTE.red), LineElement(color = PALETTE.blue)], ["Newton", "Fisher scoring"];
           position = :lb, labelsize = 12, framevisible = true, backgroundcolor = (:white, 0.9), framecolor = :white)

ax2 = Axis(fig[1, 2]; title = L"\text{Local convergence from } (2, 0.3)", xlabel = L"\text{iteration } t",
           ylabel = L"\Vert \beta^{(t)} - \hat\beta \Vert", yscale = log10)
for (method, col, mk, lab) in [(:newton, PALETTE.red, :circle, "Newton"), (:fisher, PALETTE.blue, :rect, "Fisher scoring")]
    P = iterate_fit([2.0, 0.3], method, 10)
    err = [norm(P[i, :] .- bhat) for i in 1:size(P, 1)]
    keep = err .> 1e-14
    scatterlines!(ax2, (0:10)[keep], err[keep]; color = col, marker = mk, markercolor = :white, strokecolor = col,
                  strokewidth = 1.3, markersize = 8, label = lab)
end
rho = maximum(abs.(eigvals(I - exp_info(bhat) \ obs_info(bhat))))
Pf = iterate_fit([2.0, 0.3], :fisher, 1)
e1 = norm(Pf[2, :] .- bhat)
lines!(ax2, 0:10, e1 .* rho .^ ((0:10) .- 1); color = PALETTE.grey, linestyle = :dot, label = "ρᵗ, ρ = $(round(rho; digits = 3))")
ylims!(ax2, 1e-12, 10)
axislegend(ax2; position = :lb, labelsize = 12)
colsize!(fig.layout, 1, Relative(0.53))
save_fig(fig, @__DIR__, "newton-vs-scoring")
