# Fisher geometry of N(mu, sigma^2): KL balls, and gradient versus natural gradient descent.
# Packages: CairoMakie, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Printf

set_blog_theme!()
kl(m0, s0, m, s) = log(s / s0) + (s0^2 + (m0 - m)^2) / (2 * s^2) - 0.5
xbar, s2 = 1.0, 1.0
loss(m, s) = log(s) + (s2 + (xbar - m)^2) / (2 * s^2)
grad(th) = [-(xbar - th[1]) / th[2]^2, 1 / th[2] - (s2 + (xbar - th[1])^2) / th[2]^3]
start, eta, K = [-2.0, 0.5], 0.05, 100
function descend(natural)
    th = copy(start); P = [copy(th)]
    for _ in 1:K
        d = grad(th)
        natural && (d = [th[2]^2, th[2]^2 / 2] .* d)      # F(theta)^{-1} times the gradient
        th = th - eta * d; push!(P, copy(th))
    end
    return reduce(hcat, P)'
end
Lstar = loss(xbar, sqrt(s2))
P_gd, P_ng = descend(false), descend(true)

fig = Figure(size = (780, 360))
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = L"\mu", ylabel = L"\sigma",
           title = L"\text{(a) Balls } \{\theta : \mathrm{KL}(p_{\theta_0} \Vert p_\theta) \leq 0.05\}")
mg = range(-3, 3; length = 500); sg = range(0.02, 3.2; length = 500)
for s0 in (0.25, 0.5, 1.0, 2.0), m0 in (-2.0, 0.0, 2.0)
    contour!(ax1, mg, sg, [kl(m0, s0, m, s) for m in mg, s in sg]; levels = [0.05], color = PALETTE.blue, linewidth = 1.3)
    scatter!(ax1, [m0], [s0]; color = PALETTE.ink, markersize = 4)
end
xlims!(ax1, -3, 3); ylims!(ax1, 0, 3.2)

ax2 = Axis(fig[1, 2]; xlabel = L"\mu", ylabel = L"\sigma",
           title = L"\text{(b) Fitting } N(\mu, \sigma^2)\text{, step } \eta = 0.05\text{, 100 steps}")
mg2 = range(-2.6, 2.0; length = 400); sg2 = range(0.05, 4.8; length = 400)
contour!(ax2, mg2, sg2, [loss(m, s) - Lstar for m in mg2, s in sg2];
         levels = exp.(range(log(0.01), log(8); length = 12)), color = PALETTE.grey, linewidth = 0.7)
scatterlines!(ax2, P_gd[:, 1], P_gd[:, 2]; color = PALETTE.red, markersize = 4, linewidth = 1.3,
              label = @sprintf("GD: gap %.1e", loss(P_gd[end, 1], P_gd[end, 2]) - Lstar))
scatterlines!(ax2, P_ng[:, 1], P_ng[:, 2]; color = PALETTE.blue, markersize = 4, linewidth = 1.3,
              label = @sprintf("natural GD: gap %.1e", loss(P_ng[end, 1], P_ng[end, 2]) - Lstar))
scatter!(ax2, [start[1]], [start[2]]; marker = :rect, color = PALETTE.ink, markersize = 9)
scatter!(ax2, [xbar], [1.0]; marker = :star5, color = PALETTE.ink, markersize = 15)
xlims!(ax2, -2.6, 2.0); ylims!(ax2, 0, 4.8)
axislegend(ax2; position = :rt, labelsize = 11, framevisible = true, framecolor = :white, backgroundcolor = :white)
save_fig(fig, @__DIR__, "fisher-geometry")
