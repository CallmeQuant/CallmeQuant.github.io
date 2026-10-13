# Bayesian model averaging over all 2^6 linear regressions with Zellner's g-prior (g = n):
# inclusion probabilities, the model-averaged posterior of beta_6, and out-of-sample log scores.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random, Statistics, Printf

set_blog_theme!()
const n, p = 60, 6
const beta = [0.6, 0.0, 0.4, 0.0, 0.0, 0.25]
Rm = fill(0.3, p, p); Rm[diagind(Rm)] .= 1.0; Rm[1, 2] = Rm[2, 1] = 0.8
const Lc = cholesky(Symmetric(Rm)).L
const MODELS = [BitVector(reverse(digits(i; base = 2, pad = p))) for i in 0:(2^p - 1)]

function simulate(rng, m)
    X = randn(rng, m, p) * Lc'
    return X, X * beta .+ randn(rng, m)
end

function fit(X, y, g)
    m = length(y); xbar = vec(mean(X; dims = 1)); ybar = mean(y)
    Xc = X .- xbar'; yc = y .- ybar; tss = dot(yc, yc)
    out = map(MODELS) do gam
        k = sum(gam)
        if k > 0
            Xk = Xc[:, gam]; XtXinv = inv(Xk' * Xk)
            bhat = XtXinv * (Xk' * yc); R2 = dot(yc, Xk * bhat) / tss
        else
            XtXinv = zeros(0, 0); bhat = zeros(0); R2 = 0.0
        end
        (gam = gam, logml = 0.5 * (m - 1 - k) * log1p(g) - 0.5 * (m - 1) * log1p(g * (1 - R2)),
         mean = g / (1 + g) .* bhat, S = tss * (1 - g / (1 + g) * R2), XtXinv = XtXinv, shrink = g / (1 + g))
    end
    lw = [o.logml for o in out]; w = exp.(lw .- maximum(lw))
    return out, w ./ sum(w), xbar, ybar
end

ldt(x, df, loc, scale) = logpdf(TDist(df), (x - loc) / scale) - log(scale)

function log_pred(out, xbar, ybar, Xnew, ynew, m)
    L = zeros(length(out), length(ynew))
    for (i, o) in enumerate(out)
        Xc = (Xnew .- xbar')[:, o.gam]
        loc = any(o.gam) ? ybar .+ Xc * o.mean : fill(ybar, length(ynew))
        quad = any(o.gam) ? vec(sum((Xc * o.XtXinv) .* Xc; dims = 2)) : zeros(length(ynew))
        scale = sqrt.(o.S / (m - 1) .* (1 + 1 / m .+ o.shrink .* quad))
        L[i, :] .= ldt.(ynew, m - 1, loc, scale)
    end
    return L
end

# ---- one data set
rng = MersenneTwister(7)
X, y = simulate(rng, n)
out, w, xbar, ybar = fit(X, y, n)
G = reduce(hcat, MODELS)'          # 64 x 6
pip = vec(G' * w)

grid = range(-0.3, 0.8; length = 800); j = 6
cont = zeros(length(grid))
for (o, wk) in zip(out, w)
    if o.gam[j]
        pos = findfirst(==(j), findall(o.gam))
        sd = sqrt(o.S / (n - 1) * o.shrink * o.XtXinv[pos, pos])
        cont .+= wk .* exp.(ldt.(grid, n - 1, o.mean[pos], sd))
    end
end
full = out[end]
dens_full = exp.(ldt.(grid, n - 1, full.mean[j], sqrt(full.S / (n - 1) * full.shrink * full.XtXinv[j, j])))

fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; ylabel = "posterior inclusion probability", title = "Inclusion probabilities",
           xticks = (1:p, [L"x_{%$k}" for k in 1:p]), xgridvisible = false)
barplot!(ax1, 1:p, pip; color = [b != 0 ? PALETTE.blue : PALETTE.grey for b in beta], width = 0.62)
text!(ax1, 1:p, pip .+ 0.025; text = [@sprintf("%.2f", v) for v in pip], align = (:center, :bottom), fontsize = 12)
ylims!(ax1, 0, 1.1)

ax2 = Axis(fig[1, 2]; xlabel = L"\beta_6", ylabel = "posterior density",
           title = L"\text{Model-averaged posterior of } \beta_6")
vlines!(ax2, [beta[j]]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
lines!(ax2, grid, cont; color = PALETTE.blue, label = "BMA, continuous part")
lines!(ax2, grid, dens_full; color = PALETTE.red, linestyle = :dash, linewidth = 1.8, label = "full model only")
linesegments!(ax2, [0.0, 0.0], [0.0, 1 - pip[j]]; color = PALETTE.blue, linewidth = 4)
scatter!(ax2, [0.0], [1 - pip[j]]; color = PALETTE.blue, markersize = 9)
text!(ax2, -0.27, 2.6; text = @sprintf("mass %.2f at 0", 1 - pip[j]), fontsize = 12, align = (:left, :bottom))
linesegments!(ax2, [-0.15, 0.0], [2.55, 1 - pip[j]]; color = PALETTE.ink, linewidth = 0.6)
xlims!(ax2, -0.3, 0.8); ylims!(ax2, 0, 3.6)
axislegend(ax2; position = :rt)
colsize!(fig.layout, 2, Relative(0.56))
save_fig(fig, @__DIR__, "bma-posterior")

# ---- repeated experiment
reps, m_test = 300, 200
rng = MersenneTwister(11)
diff_hpm = Float64[]; diff_full = Float64[]
for _ in 1:reps
    Xtr, ytr = simulate(rng, n); Xte, yte = simulate(rng, m_test)
    out_r, w_r, xbar_r, ybar_r = fit(Xtr, ytr, n)
    L = log_pred(out_r, xbar_r, ybar_r, Xte, yte, n)
    top = vec(maximum(L; dims = 1))
    bma = mean(log.(vec(w_r' * exp.(L .- top'))) .+ top)
    push!(diff_hpm, bma - mean(L[argmax(w_r), :]))
    push!(diff_full, bma - mean(L[end, :]))
end

fig2 = Figure(size = (780, 300))
bins = collect(-0.10:0.005:0.145)
for (k, (d, title, col)) in enumerate([(diff_hpm, "BMA minus highest-probability model", PALETTE.blue),
                                      (diff_full, "BMA minus full model", PALETTE.green)])
    ax = Axis(fig2[1, k]; title = title, xlabel = "difference in mean log predictive density",
              ylabel = k == 1 ? "replications" : "")
    hist!(ax, d; bins = bins, color = col, strokecolor = :white, strokewidth = 0.5)
    vlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 1)
    vlines!(ax, [mean(d)]; color = PALETTE.red, linestyle = :dash, linewidth = 1.6)
    text!(ax, 0.97, 0.95; space = :relative, align = (:right, :top), fontsize = 12,
          text = @sprintf("mean %.3f\npositive in %.0f%%", mean(d), 100 * mean(d .> 0)))
end
save_fig(fig2, @__DIR__, "bma-predictive")
