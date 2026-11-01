# Figures for "Markov chains, dynamic programming and fixed points" (mean-field games, Part 0).
# Packages: CairoMakie, Distributions, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics

set_blog_theme!()

# ------------------------------------------------------------------ the example MDP
const d, T, eta, kappa, lam = 11, 14, 0.3, 0.5, 0.15
const Xs = 0:(d - 1)
const As = (-1, 0, 1)
clipstate(z) = clamp(z, 0, d - 1)

p = zeros(3, d, d)                        # p[k, x+1, y+1] = p(y | x, a_k)
for (k, a) in enumerate(As), x in Xs
    for (shift, prob) in ((0, 1 - eta), (-1, eta / 2), (1, eta / 2))
        y = clipstate(x + a + shift)
        p[k, x + 1, y + 1] += prob
    end
end
cost = [kappa * abs(a) + lam * (x - 3)^2 for x in Xs, a in As]   # d x 3
g = [(x - 8.0)^2 for x in Xs]

V = zeros(T + 1, d); V[T + 1, :] = g
policy = zeros(Int, T, d)
for t in T:-1:1
    Q = hcat([cost[:, k] .+ p[k, :, :] * V[t + 1, :] for k in 1:3]...)
    for x in 1:d
        policy[t, x] = argmin(Q[x, :])
        V[t, x] = Q[x, policy[t, x]]
    end
end
m0 = zeros(d); m0[2:6] = pdf.(Binomial(4, 0.5), 0:4)
mu = zeros(T + 1, d); mu[1, :] = m0
for t in 1:T
    P = vcat([p[policy[t, x], x, :]' for x in 1:d]...)
    mu[t + 1, :] = (mu[t, :]' * P)'
end
J = sum(sum(mu[t, x] * cost[x, policy[t, x]] for x in 1:d) for t in 1:T) + sum(mu[T + 1, :] .* g)
@assert abs(J - sum(m0 .* V[1, :])) < 1e-9

# ------------------------------------------------------------------ Figure 2: backward and forward
fig = Figure(size = (760, 350))
ax1 = Axis(fig[1, 1]; title = L"(a) optimal action $\varphi_t(x)$", xlabel = L"time $t$", ylabel = L"state $x$",
           xticks = 0:2:(T - 1), yticks = 0:2:(d - 1), xgridvisible = false, ygridvisible = false)
actcolors = [PALETTE.red, colorant"#eef1f6", PALETTE.blue]
heatmap!(ax1, 0:(T - 1), Xs, Float64.(policy); colormap = actcolors, colorrange = (1, 3))
for t in 1:T, x in 1:d
    k = policy[t, x]
    if k == 2
        scatter!(ax1, [t - 1], [x - 1]; color = PALETTE.ink, markersize = 3)
    else
        scatter!(ax1, [t - 1], [x - 1]; color = :white, markersize = 8,
                 marker = k == 3 ? :utriangle : :dtriangle)
    end
end
ax2 = Axis(fig[1, 2]; title = L"(b) state distribution $\mu_t(x)$", xlabel = L"time $t$",
           xticks = 0:2:T, yticks = 0:2:(d - 1), xgridvisible = false, ygridvisible = false)
blues = cgrad([colorant"#ffffff", colorant"#c9d5ef", PALETTE.blue, colorant"#16264d"])
hm = heatmap!(ax2, 0:T, Xs, mu; colormap = blues, colorrange = (0, maximum(mu)))
Colorbar(fig[1, 3], hm; width = 10, ticklabelsize = 11)
save_fig(fig, @__DIR__, "backward-forward")

# ------------------------------------------------------------------ Figure 1: empirical measures
rng = MersenneTwister(20261101)
muT = mu[T + 1, :]
muT ./= sum(muT)                            # guard against rounding before Multinomial
fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; title = "(a) one draw of the empirical measure", xlabel = L"state $x$",
           xticks = 0:2:(d - 1), limits = (nothing, (0, 0.7)))
for (j, (N, col)) in enumerate(((40, PALETTE.gold), (400, PALETTE.green)))
    emp = rand(rng, Multinomial(N, muT)) ./ N
    barplot!(ax1, collect(Xs) .+ (j - 1.5) * 0.38, emp; width = 0.38, color = col,
             label = L"$\bar\mu^N_T$, $N = %$N$")
end
scatter!(ax1, collect(Xs), muT; color = PALETTE.ink, markersize = 8, label = L"\mu_T")
axislegend(ax1; position = :lt)

Ns = unique(round.(Int, 10 .^ range(1, 4; length = 13)))
reps = 4000
err = [mean(sum(abs.(rand(rng, Multinomial(N, muT)) ./ N .- muT)) for _ in 1:reps) for N in Ns]
clt = sqrt(2 / pi) * sum(sqrt.(muT .* (1 .- muT)))
ax2 = Axis(fig[1, 2]; title = L"(b) $L^1$ error at time $T$", xlabel = L"number of chains $N$",
           xscale = log10, yscale = log10)
scatter!(ax2, Ns, err; color = PALETTE.blue, markersize = 8,
         label = L"Monte Carlo $\mathbb{E}\Vert\bar\mu^N_T-\mu_T\Vert_1$")
lines!(ax2, Ns, sqrt.((d - 1) ./ Ns); color = PALETTE.red, label = L"bound $\sqrt{(d-1)/N}$")
lines!(ax2, Ns, clt ./ sqrt.(Ns); color = PALETTE.grey, linestyle = :dash, label = "CLT approximation")
axislegend(ax2; position = :lb)
save_fig(fig, @__DIR__, "empirical-measure")

# ------------------------------------------------------------------ Figure 3: fixed points
s = 1 / 3
fig = Figure(size = (760, 360))
ax1 = Axis(fig[1, 1]; title = "(a) three Nash equilibria", aspect = 1,
           xlabel = L"$p$ = probability that player 1 plays L", ylabel = L"$q$ = probability that player 2 plays L",
           limits = ((-0.06, 1.06), (-0.06, 1.06)))
lines!(ax1, [0, 0, 1, 1], [0, s, s, 1]; color = PALETTE.blue, linewidth = 2.6, label = L"\mathrm{BR}_1(q)")
lines!(ax1, [0, s, s, 1], [0, 0, 1, 1]; color = PALETTE.red, linewidth = 2.6, linestyle = :dash,
       label = L"\mathrm{BR}_2(p)")
scatter!(ax1, [0, s, 1], [0, s, 1]; color = :white, strokecolor = PALETTE.ink, strokewidth = 1.5, markersize = 12)
text!(ax1, 0.47, 0.17; text = L"(1/3,\,1/3)", fontsize = 14)
axislegend(ax1; position = (0.02, 0.92))

ax2 = Axis(fig[1, 2]; title = "(b) why the values must be convex", aspect = 1,
           xlabel = L"x", ylabel = L"F(x),\ G(x)", limits = ((-0.03, 1.03), (-0.03, 1.03)))
lines!(ax2, [0, 1], [0, 1]; color = PALETTE.grey, linestyle = :dot)
lines!(ax2, [0, 0.5], [0.8, 0.8]; color = PALETTE.blue, linewidth = 2.6, label = L"F(1/2) = [0.2, 0.8]")
lines!(ax2, [0.5, 1], [0.2, 0.2]; color = PALETTE.blue, linewidth = 2.6)
lines!(ax2, [0.5, 0.5], [0.2, 0.8]; color = PALETTE.blue, linewidth = 2.6)
scatter!(ax2, [0.5, 0.5], [0.2, 0.8]; color = PALETTE.red, markersize = 10, label = L"G(1/2) = \{0.2, 0.8\}")
scatter!(ax2, [0.5], [0.5]; color = :white, strokecolor = PALETTE.ink, strokewidth = 1.5, markersize = 12)
text!(ax2, 0.55, 0.45; text = L"fixed point of $F$", fontsize = 13)
axislegend(ax2; position = :lt)
save_fig(fig, @__DIR__, "fixed-points")

println("J = ", round(J; digits = 6), ",  <m0, V0> = ", round(sum(m0 .* V[1, :]); digits = 6))
