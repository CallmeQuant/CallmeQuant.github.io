# Symmetric Beta priors for a coin's head probability and the posteriors after five heads in five
# tosses, with posterior modes (note: maximum a posteriori estimation).
# Packages: CairoMakie, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions

set_blog_theme!()
t = range(0.0005, 0.9995; length = 800)
alphas = [1, 2, 5, 10]
n1, n2 = 5, 0

fig = Figure(size = (760, 320))
ax1 = Axis(fig[1, 1]; xlabel = L"\theta", ylabel = "density", title = L"\text{Priors } \mathrm{Beta}(\alpha, \alpha)")
for (k, a) in enumerate(alphas)
    lines!(ax1, t, pdf.(Beta(a, a), t); color = CYCLE[k], label = L"\alpha=\beta=%$a")
end
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; xlabel = L"\theta", title = "Posteriors after 5 heads in 5 tosses; dots: MAP")
vlines!(ax2, [1.0]; color = PALETTE.grey, linewidth = 1, linestyle = :dash)
text!(ax2, 0.985, 6.25; text = "MLE", align = (:right, :center), fontsize = 12, color = PALETTE.grey)
for (k, a) in enumerate(alphas)
    pa, pb = n1 + a, n2 + a
    lines!(ax2, t, pdf.(Beta(pa, pb), t); color = CYCLE[k], label = L"\mathrm{Beta}(%$pa, %$pb)")
    mode = (pa - 1) / (pa + pb - 2)
    scatter!(ax2, [mode], [pdf(Beta(pa, pb), min(mode, 0.9995))]; color = CYCLE[k], markersize = 9)
end
axislegend(ax2; position = :lt)
linkyaxes!(ax1, ax2)
hideydecorations!(ax2; grid = false)
for ax in (ax1, ax2)
    xlims!(ax, 0, 1); ylims!(ax, 0, 6.6)
end
save_fig(fig, @__DIR__, "beta-densities")
