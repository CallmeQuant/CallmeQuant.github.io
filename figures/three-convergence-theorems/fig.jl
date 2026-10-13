# Escaping mass and domination.
# (a) f_n = n 1_(0,1/n]; (b) f_n = 1_[n,n+1); (c) f_n(x) = (1 + x/n)^(-n), dominated by (1 + x/2)^(-2).
# Packages: CairoMakie
include(joinpath(@__DIR__, "..", "_style", "style.jl"))

set_blog_theme!()
ns = [1, 2, 4, 8]
cols = [PALETTE.gold, PALETTE.green, PALETTE.red, PALETTE.blue]
fig = Figure(size = (860, 300))

ax1 = Axis(fig[1, 1]; xlabel = L"x", title = L"\text{(a) } f_n = n\,\mathbf{1}_{(0, 1/n]}")
for (n, c) in zip(ns, cols)
    lines!(ax1, [0, 0, 1 / n, 1 / n, 1.2], [0, n, n, 0, 0]; color = c, label = L"n = %$n")
end
xlims!(ax1, -0.05, 1.2); ylims!(ax1, 0, 9)
axislegend(ax1; position = :rt)

ax2 = Axis(fig[1, 2]; xlabel = L"x", title = L"\text{(b) } f_n = \mathbf{1}_{[n, n+1)}")
for (n, c) in zip(ns, cols)
    lines!(ax2, [0, n, n, n + 1, n + 1, 10], [0, 0, 1, 1, 0, 0]; color = c)
end
xlims!(ax2, 0, 10); ylims!(ax2, 0, 1.4)

x = range(0, 6; length = 400)
ax3 = Axis(fig[1, 3]; xlabel = L"x", title = L"\text{(c) } f_n = (1 + x/n)^{-n},\ n \geq 2")
band!(ax3, x, zeros(length(x)), (1 .+ x ./ 2) .^ -2; color = PALETTE.light, label = L"g = (1 + x/2)^{-2}")
for (n, c) in zip(ns[2:end], cols[2:end])
    lines!(ax3, x, (1 .+ x ./ n) .^ -n; color = c)
end
lines!(ax3, x, exp.(-x); color = PALETTE.ink, linestyle = :dash, linewidth = 1.5, label = L"e^{-x}")
xlims!(ax3, 0, 6); ylims!(ax3, 0, 1.05)
axislegend(ax3; position = :rt)
save_fig(fig, @__DIR__, "escaping-mass")
