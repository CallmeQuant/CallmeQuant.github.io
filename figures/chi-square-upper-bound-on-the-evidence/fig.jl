# ELBO and chi-square upper bound (CUBO_2) for a skew-normal posterior: optimal Gaussian fits, the
# evidence sandwich, and Monte Carlo behaviour of the CUBO estimate.
# Packages: CairoMakie, Distributions, QuadGK, Optim, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, QuadGK, Optim, Random, Statistics

set_blog_theme!()
const ALPHA = 3.0
const LOG_EV = log(0.5)
log_joint(z) = logcdf(Normal(), ALPHA * z) + logpdf(Normal(), z)

elbo(m, s) = quadgk(t -> pdf(Normal(m, s), t) * (log_joint(t) - logpdf(Normal(m, s), t)), m - 12s, m + 12s)[1]
function cubo2(m, s)
    (s^2 < 0.5 || (s^2 == 0.5 && m <= 0)) && return Inf
    f(t) = exp(2 * log_joint(t) - logpdf(Normal(m, s), t) - 2 * LOG_EV)
    return LOG_EV + 0.5 * log(quadgk(f, -Inf, m)[1] + quadgk(f, m, Inf)[1])
end

re = optimize(v -> -elbo(v[1], exp(v[2])), [0.75, log(0.6)], NelderMead())
rc = optimize(v -> cubo2(v[1], exp(v[2])), [0.75, log(0.75)], NelderMead())
m_e, s_e = Optim.minimizer(re)[1], exp(Optim.minimizer(re)[2])
m_c, s_c = Optim.minimizer(rc)[1], exp(Optim.minimizer(rc)[2])

# ---------- Figure 1 ----------
fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"z", ylabel = "density", title = "Posterior and Gaussian fits")
z = collect(range(-2.2, 3.4; length = 600))
post = exp.(log_joint.(z) .- LOG_EV)
band!(ax1, z, zero(z), post; color = PALETTE.light)
lines!(ax1, z, post; color = PALETTE.ink, linewidth = 1.6, label = L"\text{posterior } \pi")
lines!(ax1, z, pdf.(Normal(m_e, s_e), z); color = PALETTE.blue, label = L"\arg\max \, \mathrm{ELBO}")
lines!(ax1, z, pdf.(Normal(m_c, s_c), z); color = PALETTE.red, label = L"\arg\min \, \mathrm{CUBO}_2")
xlims!(ax1, -2.2, 3.4); ylims!(ax1, 0, 1.0)
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{scale } s \text{ of } q = \mathcal{N}(m, s^2)", ylabel = "nats",
           title = "The evidence sandwich")
sg = collect(range(0.45, 1.3; length = 70))
el = [-Optim.minimum(optimize(m -> -elbo(m, s), 0.2, 1.3)) for s in sg]
sgc = vcat(sqrt(0.5), collect(range(sqrt(0.5) + 1e-4, 1.3; length = 70)))
cu = [Optim.minimum(optimize(m -> cubo2(m, s), 0.05, 1.5)) for s in sgc]
vspan!(ax2, 0.45, sqrt(0.5); color = PALETTE.light)
text!(ax2, 0.465, -0.60; text = L"\mathrm{CUBO}_2 = +\infty \text{ for } s^2 < 1/2", align = (:left, :top), fontsize = 11)
hlines!(ax2, [LOG_EV]; color = PALETTE.ink, linestyle = :dash, linewidth = 1.2)
text!(ax2, 1.29, LOG_EV + 0.004; text = L"\log p(y)", align = (:right, :bottom), fontsize = 12)
lines!(ax2, sg, el; color = PALETTE.blue, label = L"\max_m \mathrm{ELBO}")
lines!(ax2, sgc, cu; color = PALETTE.red, label = L"\min_m \mathrm{CUBO}_2")
scatter!(ax2, [s_e, s_c], [elbo(m_e, s_e), cubo2(m_c, s_c)]; color = [PALETTE.blue, PALETTE.red], markersize = 9)
xlims!(ax2, 0.45, 1.3); ylims!(ax2, -0.95, -0.56)
axislegend(ax2; position = :rb)
save_fig(fig, @__DIR__, "fits-and-sandwich")

# ---------- Figure 2 ----------
rng = MersenneTwister(2017)
S_list = [10, 30, 100, 300, 1000, 3000, 10000]
reps = 2000
function cubo_hat(m, s, S)
    out = zeros(reps)
    for k in 1:reps
        zz = m .+ s .* randn(rng, S)
        lw = 2 .* (log_joint.(zz) .- logpdf.(Normal(m, s), zz))
        mx = maximum(lw)
        out[k] = 0.5 * (mx + log(mean(exp.(lw .- mx))))
    end
    return out
end
fig = Figure(size = (780, 320))
cases = [(m_c, s_c, PALETTE.red, "q = argmin CUBO₂ (s² = $(round(s_c^2; digits = 3)))", cubo2(m_c, s_c)),
         (m_e, s_e, PALETTE.blue, "q = argmax ELBO (s² = $(round(s_e^2; digits = 3)))", Inf)]
axs = Axis[]
for (j, (m, s, col, ttl, truth)) in enumerate(cases)
    ax = Axis(fig[1, j]; xscale = log10, title = ttl, xlabel = L"\text{number of draws } S",
              ylabel = j == 1 ? L"\text{estimate of } \mathrm{CUBO}_2(q)" : "")
    qs = [quantile(cubo_hat(m, s, S), [0.1, 0.5, 0.9]) for S in S_list]
    band!(ax, S_list, getindex.(qs, 1), getindex.(qs, 3); color = (col, 0.18), label = "10%–90% range")
    scatterlines!(ax, S_list, getindex.(qs, 2); color = col, markersize = 7, label = "median")
    hlines!(ax, [LOG_EV]; color = PALETTE.ink, linestyle = :dash, linewidth = 1.2, label = L"\log p(y)")
    isfinite(truth) && hlines!(ax, [truth]; color = col, linestyle = :dot, linewidth = 1.2, label = L"\text{true } \mathrm{CUBO}_2(q)")
    ylims!(ax, -0.86, -0.42)
    axislegend(ax; position = :rb, labelsize = 11)
    push!(axs, ax)
end
linkyaxes!(axs...)
hideydecorations!(axs[2]; grid = false)
save_fig(fig, @__DIR__, "monte-carlo-cubo")
