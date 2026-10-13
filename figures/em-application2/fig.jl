# EM versus stochastic EM (SEM) for a two-component univariate Gaussian mixture.
# (a) trajectories of the ordered component means; (b) SEM iterates of the larger mean after burn-in.
# Packages: CairoMakie, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, Statistics

set_blog_theme!()
rng = MersenneTwister(113)
n = 400
z = rand(rng, n) .< 0.35
x = [zi ? 2.0 + 0.8 * randn(rng) : -1.0 + 1.2 * randn(rng) for zi in z]

function resp(w, mu, s2)
    lp = hcat([log(w[k]) .- 0.5 * log(2pi * s2[k]) .- 0.5 .* (x .- mu[k]) .^ 2 ./ s2[k] for k in 1:2]...)
    m = maximum(lp; dims = 2); e = exp.(lp .- m)
    return e ./ sum(e; dims = 2)
end
function m_step(g)
    Nk = vec(sum(g; dims = 1))
    mu = vec(sum(g .* x; dims = 1)) ./ Nk
    s2 = vec(sum(g .* (x .- mu') .^ 2; dims = 1)) ./ Nk
    return Nk ./ n, mu, s2
end

T, burn = 3000, 300
start = ([0.5, 0.5], [-0.01, 0.01], [3.0, 3.0])
em = zeros(T + 1, 2); sem = zeros(T + 1, 2)
p = start; em[1, :] = sort(p[2])
for t in 1:T
    global p = m_step(resp(p...))
    em[t+1, :] = sort(p[2])
end
p = start; sem[1, :] = sort(p[2])
for t in 1:T
    g = resp(p...)
    oh = zeros(n, 2)
    while true                        # S-step; redraw if a component gets fewer than 2 points
        lab = [rand(rng) < g[i, 1] ? 1 : 2 for i in 1:n]
        oh = hcat(Float64.(lab .== 1), Float64.(lab .== 2))
        minimum(sum(oh; dims = 1)) >= 2 && break
    end
    global p = m_step(oh)
    sem[t+1, :] = sort(p[2])          # order the components by their means
end

k = 500
fig = Figure(size = (800, 370))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{iteration } t", title = "(a) component means")
lines!(ax1, 0:k, sem[1:k+1, 1]; color = PALETTE.blue, linewidth = 1, label = L"\text{SEM } \mu_{(1)}")
lines!(ax1, 0:k, em[1:k+1, 1]; color = PALETTE.ink, linewidth = 2, label = L"\text{EM } \mu_{(1)}")
lines!(ax1, 0:k, sem[1:k+1, 2]; color = PALETTE.red, linewidth = 1, label = L"\text{SEM } \mu_{(2)}")
lines!(ax1, 0:k, em[1:k+1, 2]; color = PALETTE.ink, linewidth = 2, linestyle = :dash, label = L"\text{EM } \mu_{(2)}")
Legend(fig[2, 1], ax1; orientation = :horizontal)

draws = sem[burn+2:end, 2]
ax2 = Axis(fig[1, 2]; xlabel = L"\mu_{(2)}", ylabel = "density", title = L"\text{(b) SEM iterates},\ t > 300")
hist!(ax2, draws; bins = 40, normalization = :pdf, color = (PALETTE.red, 0.35))
vlines!(ax2, [em[end, 2]]; color = PALETTE.ink, linewidth = 2, label = "EM limit")
vlines!(ax2, [mean(draws)]; color = PALETTE.red, linewidth = 2, linestyle = :dash, label = "SEM average")
Legend(fig[2, 2], ax2; orientation = :horizontal)
colsize!(fig.layout, 1, Relative(0.6))
save_fig(fig, @__DIR__, "em-vs-sem")
