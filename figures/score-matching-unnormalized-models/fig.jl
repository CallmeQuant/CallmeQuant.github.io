# Score matching for the unnormalized double-well family log q(x) = -t1 x^2/2 - t2 x^4/4,
# compared with maximum likelihood (Newton's method with the normalizer computed on a grid).
# Packages: CairoMakie, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random, Statistics, Printf

set_blog_theme!()
theta0 = [-2.0, 1.0]
logq(x, t) = -t[1] * x^2 / 2 - t[2] * x^4 / 4
grid = range(-5, 5; length = 20001); dx = step(grid)

function sample_q(n, rng)                      # exact draws by rejection from N(0, 1.5^2)
    s = 1.5; logM = maximum(logq.(grid, Ref(theta0)) .+ grid .^ 2 ./ (2s^2))
    out = Float64[]
    while length(out) < n
        x = s .* randn(rng, 2n)
        keep = log.(rand(rng, 2n)) .< logq.(x, Ref(theta0)) .+ x .^ 2 ./ (2s^2) .- logM
        append!(out, x[keep])
    end
    return out[1:n]
end

function score_matching(x)
    A = [mean(x .^ 2) mean(x .^ 4); mean(x .^ 4) mean(x .^ 6)]
    return -(A \ [-1.0, -3mean(x .^ 2)])
end

function log_Z(t)
    v = logq.(grid, Ref(t)); m = maximum(v)
    return m + log(sum(exp.(v .- m)) * dx)
end

function mle(x, t)                              # Newton iterations on the exponential-family likelihood
    S = [mean(x .^ 2) / 2, mean(x .^ 4) / 4]   # sufficient statistics (sign absorbed below)
    nll(t) = t[2] > 0 ? dot(t, S) + log_Z(t) : Inf
    for _ in 1:50
        w = exp.(logq.(grid, Ref(t)) .- log_Z(t)) .* dx
        U = hcat(grid .^ 2 ./ 2, grid .^ 4 ./ 4)
        EU = U' * w
        g = S .- EU                              # gradient of the negative log-likelihood
        H = (U .- EU')' * ((U .- EU') .* w)       # covariance of U under the model
        step_ = H \ g; a = 1.0
        while nll(t .- a .* step_) > nll(t) && a > 1e-8
            a /= 2
        end
        t = t .- a .* step_
        norm(g) < 1e-10 && break
    end
    return t
end

rng = MersenneTwister(2005)
fig = Figure(size = (780, 340))
x = sample_q(1000, rng)
t_sm = score_matching(x); t_ml = mle(x, t_sm)
xs = range(-3.2, 3.2; length = 600)
ax = Axis(fig[1, 1]; xlabel = L"x", ylabel = "density",
          title = L"n = 1000 \text{ draws from } q(x) \propto e^{x^2 - x^4/4}")
hist!(ax, x; bins = range(-3.2, 3.2; length = 50), normalization = :pdf, color = PALETTE.light,
      strokecolor = :white, strokewidth = 0.5)
for (t, col, lab, ls) in [(theta0, PALETTE.ink, "true (-2, 1)", :solid),
                          (t_sm, PALETTE.blue, @sprintf("score matching (%.2f, %.2f)", t_sm...), :solid),
                          (t_ml, PALETTE.red, @sprintf("maximum likelihood (%.2f, %.2f)", t_ml...), :dash)]
    lines!(ax, xs, exp.(logq.(xs, Ref(t)) .- log_Z(t)); color = col, linestyle = ls, label = lab)
end
xlims!(ax, -3.2, 3.2); ylims!(ax, 0, 0.62)
axislegend(ax; position = :ct, labelsize = 11)

R, n = 400, 500
est_sm = zeros(R, 2); est_ml = zeros(R, 2)
for r in 1:R
    xr = sample_q(n, rng)
    est_sm[r, :] = score_matching(xr)
    est_ml[r, :] = mle(xr, est_sm[r, :])
end
ax = Axis(fig[1, 2]; xlabel = L"\hat\theta_1", ylabel = L"\hat\theta_2", title = "$R data sets of size n = $n")
ang = range(0, 2pi; length = 200)
for (est, col, lab) in [(est_sm, PALETTE.blue, "score matching"), (est_ml, PALETTE.red, "maximum likelihood")]
    scatter!(ax, est[:, 1], est[:, 2]; color = (col, 0.5), markersize = 4, label = lab)
    C = cholesky(Symmetric(cov(est))).L          # 95% normal-theory ellipse
    E = mean(est; dims = 1) .+ sqrt(5.991) .* hcat(cos.(ang), sin.(ang)) * C'
    lines!(ax, E[:, 1], E[:, 2]; color = col, linewidth = 1.6)
end
scatter!(ax, [theta0[1]], [theta0[2]]; marker = :cross, markersize = 16, color = PALETTE.ink)
axislegend(ax; position = :lb, labelsize = 11)
save_fig(fig, @__DIR__, "double-well")
