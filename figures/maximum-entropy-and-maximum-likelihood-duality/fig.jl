# Duality between maximum entropy and maximum likelihood on X = {0,1,2,3}, T(x) = x.
# Packages: CairoMakie, Distributions, Random, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Printf

set_blog_theme!()
rng = MersenneTwister(7)
x = collect(0:3)
true_p = [0.15, 0.45, 0.10, 0.30]
n = 30
dat = rand(rng, Categorical(true_p), n) .- 1
p_hat = [count(==(v), dat) for v in x] ./ n
t_hat = sum(dat) / n

A(th) = log(sum(exp.(th .* x)))
mean_of(th) = sum(x .* exp.(th .* x)) / sum(exp.(th .* x))
pmf(th) = exp.(th .* x .- A(th))
H(p) = -sum(q > 0 ? q * log(q) : 0.0 for q in p)
function solve_theta(m; lo = -40.0, hi = 40.0)      # bisection; mean_of is increasing
    for _ in 1:200
        mid = (lo + hi) / 2
        mean_of(mid) < m ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
theta_hat = solve_theta(t_hat)
g_min = A(theta_hat) - theta_hat * t_hat

fig = Figure(size = (980, 320))
ax1 = Axis(fig[1, 1]; xlabel = L"\theta", title = L"\text{Dual: } A(\theta) - \theta\,\hat t = -\ell_n(\theta)")
th = range(-2.5, 2.0; length = 400)
lines!(ax1, th, A.(th) .- th .* t_hat; color = PALETTE.blue)
hlines!(ax1, [H(p_hat)]; color = PALETTE.red, linestyle = :dash, linewidth = 1.4)
hlines!(ax1, [g_min]; color = PALETTE.ink, linestyle = :dot, linewidth = 0.8)
scatter!(ax1, [theta_hat], [g_min]; color = PALETTE.ink, markersize = 9)
text!(ax1, theta_hat + 0.05, g_min - 0.04; text = L"\hat\theta", align = (:left, :top), fontsize = 14)
text!(ax1, -2.4, H(p_hat) - 0.05; text = L"H(\hat p_n)", color = PALETTE.red, align = (:left, :top), fontsize = 14)
xlims!(ax1, -2.5, 2.0); ylims!(ax1, 0.9, 2.6)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{mean } m", ylabel = "entropy (nats)",
           title = L"\text{Primal: lines } m \mapsto A(\theta) - \theta m")
ms = range(0.02, 2.98; length = 300)
for t0 in [-1.5, -0.6, 0.7, 1.6]
    lines!(ax2, ms, A(t0) .- t0 .* ms; color = PALETTE.grey, linewidth = 1)
end
lines!(ax2, ms, A(theta_hat) .- theta_hat .* ms; color = PALETTE.ink, linestyle = :dash, linewidth = 1)
lines!(ax2, ms, [H(pmf(solve_theta(m))) for m in ms]; color = PALETTE.blue, linewidth = 2.6)
scatter!(ax2, [t_hat], [H(pmf(theta_hat))]; color = PALETTE.ink, markersize = 9)
scatter!(ax2, [t_hat], [H(p_hat)]; color = PALETTE.red, markersize = 9)
text!(ax2, t_hat + 0.07, H(p_hat); text = L"\hat p_n", color = PALETTE.red, align = (:left, :center), fontsize = 14)
text!(ax2, t_hat - 0.05, H(pmf(theta_hat)) + 0.07; text = L"p_{\hat\theta}", align = (:right, :bottom), fontsize = 14)
xlims!(ax2, 0, 3); ylims!(ax2, 0, 1.75)

ax3 = Axis(fig[1, 3]; xlabel = L"x", ylabel = "probability", xticks = 0:3, xgridvisible = false,
           title = @sprintf("Same mean t̂ = %.2f", t_hat))
w = 0.38
barplot!(ax3, x .- w / 2, p_hat; width = w, color = PALETTE.red, label = L"\hat p_n")
barplot!(ax3, x .+ w / 2, pmf(theta_hat); width = w, color = PALETTE.blue, label = L"p_{\hat\theta}")
ylims!(ax3, 0, 0.62)
axislegend(ax3; position = :lt, orientation = :horizontal)
colsize!(fig.layout, 3, Relative(0.27))
save_fig(fig, @__DIR__, "duality")
