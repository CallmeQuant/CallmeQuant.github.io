# Stopping rules and the likelihood principle: three successes in twelve Bernoulli trials under a
# fixed-n (binomial) design and a sample-until-third-success (negative binomial) design, and the
# common posterior under a uniform prior.
# Packages: CairoMakie, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions

set_blog_theme!()
fig = Figure(size = (840, 300))

k = 0:12
pk = pdf.(Binomial(12, 0.5), k)
ax1 = Axis(fig[1, 1]; xlabel = L"\text{successes } X \text{ in } n = 12", ylabel = L"\text{probability under } \theta = 1/2",
           title = L"\text{Binomial: } P(X \leq 3) = 0.073", xticks = [0, 3, 6, 9, 12], xgridvisible = false)
barplot!(ax1, collect(k), pk; color = [x <= 3 ? PALETTE.red : PALETTE.light for x in k], width = 0.75)
ylims!(ax1, 0, 0.25)

n = 3:25
pn = [binomial(m - 1, 2) * 0.5^m for m in n]
ax2 = Axis(fig[1, 2]; xlabel = L"\text{trial } N \text{ of the third success}",
           title = L"\text{Neg. binomial: } P(N \geq 12) = 0.033", xticks = 3:3:24, xgridvisible = false)
barplot!(ax2, collect(n), pn; color = [m >= 12 ? PALETTE.red : PALETTE.light for m in n], width = 0.75)
ylims!(ax2, 0, 0.25)

t = range(0, 1; length = 600)
d = pdf.(Beta(4, 10), t)
ax3 = Axis(fig[1, 3]; xlabel = L"\theta", ylabel = "posterior density", title = L"\text{Posterior: Beta}(4, 10)")
tt = t[t .>= 0.5]
band!(ax3, collect(tt), zeros(length(tt)), pdf.(Beta(4, 10), tt); color = PALETTE.red)
lines!(ax3, t, d; color = PALETTE.blue)
linesegments!(ax3, [0.53, 0.62], [0.25, 1.5]; color = PALETTE.ink, linewidth = 0.7)
text!(ax3, 0.5, 1.6; text = L"P(\theta \geq 1/2) = 0.046", fontsize = 13, align = (:left, :bottom))
xlims!(ax3, 0, 1); ylims!(ax3, 0, 3.6)
save_fig(fig, @__DIR__, "stopping-rule")
