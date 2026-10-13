# Improper priors: collapse of Beta(eps, eps) posteriors when X = 0, and the posterior of log(tau)
# in a normal hierarchical model under the priors 1/tau and 1.
# Packages: CairoMakie, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions

set_blog_theme!()

n = 10
t = range(0, 0.5; length = 600)
eps_values = [1.0, 0.3, 0.1, 0.03, 0.01]

y = [2.1, -0.4, 1.3, 0.2, 3.0, -1.1, 0.9, 1.7]
s = [1.0, 0.8, 1.5, 1.2, 1.0, 2.0, 0.9, 1.1]
function log_marg(tau)
    w = 1 ./ (s .^ 2 .+ tau^2)
    W = sum(w)
    mu = sum(w .* y) / W
    J = length(y)
    return -0.5 * (J - 1) * log(2pi) - 0.5 * log(W) + 0.5 * sum(log.(w)) - 0.5 * sum(w .* (y .- mu) .^ 2)
end
logtau = range(log(1e-4), log(1e2); length = 800)
lm = log_marg.(exp.(logtau))
lu = lm .+ logtau
x = logtau ./ log(10)

fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"t", ylabel = L"P(\theta \leq t \mid X = 0)",
           title = L"\mathrm{Beta}(\varepsilon,\varepsilon)\ \text{prior},\ n = 10,\ X = 0")
for (k, e) in enumerate(eps_values)
    lines!(ax1, t, cdf.(Beta(e, n + e), t); color = CYCLE[k], label = L"\varepsilon = %$e")
end
xlims!(ax1, 0, 0.5); ylims!(ax1, 0, 1.03)
axislegend(ax1; position = :rb)

ax2 = Axis(fig[1, 2]; xlabel = L"\tau\ \text{(log scale)}", ylabel = L"\text{density of } \log\tau \text{ (unnormalized)}",
           title = L"\text{Hierarchical model},\ J = 8\ \text{groups}",
           xticks = (-4:2, [L"10^{%$k}" for k in -4:2]))
lines!(ax2, x, exp.(lm .- maximum(lm)); color = PALETTE.red, label = L"\pi(\tau) \propto 1/\tau")
lines!(ax2, x, exp.(lu .- maximum(lu)); color = PALETTE.blue, label = L"\pi(\tau) \propto 1")
xlims!(ax2, -4, 2); ylims!(ax2, 0, 1.08)
axislegend(ax2; position = :lt)
save_fig(fig, @__DIR__, "improper-limits")
