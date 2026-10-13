# Lasso sign recovery and the irrepresentable condition.
# Design: X1, X2 independent N(0,1); X3 has correlation c with each; beta* = (1, 1, 0); 2c is the
# irrepresentable quantity, so the condition holds for c = 0.4 and fails for c = 0.65.
# Packages: CairoMakie, Random, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, LinearAlgebra, Statistics

set_blog_theme!()
rng = MersenneTwister(2006)
const SIGMA, B = 1.0, 1.0

function design(n, c; p_extra = 0)
    x1, x2 = randn(rng, n), randn(rng, n)
    x3 = c .* x1 .+ c .* x2 .+ sqrt(1 - 2c^2) .* randn(rng, n)
    X = hcat(x1, x2, x3, randn(rng, n, p_extra))
    return X, B .* (x1 .+ x2) .+ SIGMA .* randn(rng, n)
end

soft(u, t) = sign(u) * max(abs(u) - t, 0.0)

# Coordinate descent for (1/2n)||y - X b||^2 + lambda ||b||_1, warm-started from b.
function lasso_cd!(b, X, y, lambda; tol = 1e-10, maxit = 10_000)
    n = size(X, 1)
    colsq = vec(sum(abs2, X; dims = 1)) ./ n
    r = y .- X * b
    for _ in 1:maxit
        maxchange = 0.0
        for j in eachindex(b)
            xj = view(X, :, j)
            old = b[j]
            new = soft(dot(xj, r) / n + colsq[j] * old, lambda) / colsq[j]
            if new != old
                r .-= xj .* (new - old)
                b[j] = new
                maxchange = max(maxchange, abs(new - old))
            end
        end
        maxchange < tol && break
    end
    return b
end

cases = [(0.4, L"c = 0.4:\ \Vert \Gamma z_S \Vert_\infty = 0.8"), (0.65, L"c = 0.65:\ \Vert \Gamma z_S \Vert_\infty = 1.3")]

# ---- Figure 1 ----
lams = 10 .^ range(0.3, -3; length = 200)
fig = Figure(size = (780, 330))
axs = Axis[]
for (k, (c, ttl)) in enumerate(cases)
    X, y = design(1000, c)
    b = zeros(3); path = zeros(length(lams), 3)
    for (i, lam) in enumerate(lams)
        path[i, :] = lasso_cd!(b, X, y, lam)
    end
    ax = Axis(fig[1, k]; title = ttl, xscale = log10, xreversed = true, titlesize = 14,
              xlabel = L"\lambda\ \text{(log scale, decreasing)}", ylabel = k == 1 ? "lasso coefficient" : "")
    hlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.8)
    lines!(ax, lams, path[:, 1]; color = PALETTE.blue, label = L"\hat\beta_1")
    lines!(ax, lams, path[:, 2]; color = PALETTE.green, linestyle = :dash, label = L"\hat\beta_2")
    lines!(ax, lams, path[:, 3]; color = PALETTE.red, label = L"\hat\beta_3\ \text{(true value 0)}")
    k == 1 && axislegend(ax; position = :rc, labelsize = 12)
    push!(axs, ax)
end
linkyaxes!(axs...)
save_fig(fig, @__DIR__, "lasso-paths")

# ---- Figure 2 ----
ns = [100, 200, 400, 800, 1600, 3200, 6400]
R, p = 200, 50
truth = sign.(vcat(B, B, zeros(p - 2)))
fig2 = Figure(size = (640, 330))
ax = Axis(fig2[1, 1]; xscale = log10, xticks = (ns, string.(ns)), xlabel = L"\text{sample size } n \text{ (log scale)}",
          ylabel = L"\mathrm{P}\{\mathrm{sign}(\hat\beta) = \mathrm{sign}(\beta^*)\}")
for ((c, ttl), col, mk) in zip(cases, [PALETTE.blue, PALETTE.red], [:circle, :rect])
    prob = Float64[]
    for n in ns
        lam = 4 * SIGMA * sqrt(2 * log(p) / n)
        hits = 0
        for _ in 1:R
            X, y = design(n, c; p_extra = p - 3)
            b = lasso_cd!(zeros(p), X, y, lam)
            b[abs.(b) .< 1e-10] .= 0
            hits += sign.(b) == truth
        end
        push!(prob, hits / R)
    end
    scatterlines!(ax, ns, prob; color = col, marker = mk, markercolor = :white, strokecolor = col,
                  strokewidth = 1.4, markersize = 9, label = ttl)
end
ylims!(ax, -0.03, 1.03)
axislegend(ax; position = :rc, labelsize = 12)
save_fig(fig2, @__DIR__, "sign-recovery")
