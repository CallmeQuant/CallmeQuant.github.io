# Binomial (Cox-Ross-Rubinstein) option pricing: a four-period American-put tree with
# early-exercise nodes, CRR call error against Black-Scholes, American versus European put.
# Packages: CairoMakie, Distributions, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Printf

set_blog_theme!()

# ---------------------------------------------------------------- tree
S0, K, u, R, N = 100.0, 100.0, 1.15, 1.01, 4
d = 1 / u
q = (R - d) / (u - d)
S = Dict((n, k) => S0 * u^k * d^(n - k) for n in 0:N for k in 0:n)
V = Dict{Tuple{Int,Int},Float64}(); ex = Dict{Tuple{Int,Int},Bool}()
for k in 0:N
    V[(N, k)] = max(K - S[(N, k)], 0.0); ex[(N, k)] = false
end
for n in N-1:-1:0, k in 0:n
    cont = (q * V[(n + 1, k + 1)] + (1 - q) * V[(n + 1, k)]) / R
    stp = max(K - S[(n, k)], 0.0)
    V[(n, k)] = max(cont, stp); ex[(n, k)] = stp > cont
end

fig = Figure(size = (720, 400))
ax = Axis(fig[1, 1]; xlabel = L"\text{period } n", xticks = 0:N,
          title = L"\text{American put, } K = 100,\ u = 1.15,\ d = 1/u,\ R = 1.01:\ \text{node shows } S_n \text{ (top) and } V_n \text{ (bottom)}",
          titlesize = 13, xgridvisible = false, ygridvisible = false,
          leftspinevisible = false, yticksvisible = false, yticklabelsvisible = false)
for n in 0:N-1, k in 0:n, kk in (k, k + 1)
    lines!(ax, [n, n + 1], [2k - n, 2kk - n - 1]; color = PALETTE.light, linewidth = 2)
end
for n in 0:N, k in 0:n
    x, y = n, 2k - n
    hit = ex[(n, k)]
    col = hit ? PALETTE.red : PALETTE.blue
    scatter!(ax, [x], [y]; markersize = 52, color = hit ? col : :white, strokecolor = col, strokewidth = 1.5)
    tc = hit ? :white : PALETTE.ink
    text!(ax, x, y + 0.2; text = @sprintf("%.1f", S[(n, k)]), align = (:center, :center), fontsize = 12, color = tc)
    text!(ax, x, y - 0.22; text = @sprintf("%.2f", V[(n, k)]), align = (:center, :center), fontsize = 12,
          color = tc, font = :bold)
end
xlims!(ax, -0.5, N + 0.5); ylims!(ax, -N - 0.8, N + 0.8)
elems = [MarkerElement(marker = :circle, color = PALETTE.red, strokecolor = PALETTE.red, markersize = 12),
         MarkerElement(marker = :circle, color = :white, strokecolor = PALETTE.blue, strokewidth = 1.5, markersize = 12)]
axislegend(ax, elems, ["early exercise optimal", "continue (or expiry)"]; position = :lt)
save_fig(fig, @__DIR__, "tree")

# ------------------------------------------------------------ convergence
function crr(S0, K, r, sigma, T, N; kind = :call, american = false)
    dt = T / N; u = exp(sigma * sqrt(dt)); d = 1 / u; R = exp(r * dt)
    q = (R - d) / (u - d)
    pay = kind == :call ? (s -> max(s - K, 0.0)) : (s -> max(K - s, 0.0))
    V = [pay(S0 * u^k * d^(N - k)) for k in 0:N]
    for n in N-1:-1:0
        V = (q .* V[2:n+2] .+ (1 - q) .* V[1:n+1]) ./ R
        if american
            V = max.(V, [pay(S0 * u^k * d^(n - k)) for k in 0:n])
        end
    end
    return V[1]
end

function bs(S0, K, r, sigma, T; kind = :call)
    d1 = (log(S0 / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T)); d2 = d1 - sigma * sqrt(T)
    Phi(x) = cdf(Normal(), x)
    return kind == :call ? S0 * Phi(d1) - K * exp(-r * T) * Phi(d2) :
                           K * exp(-r * T) * Phi(-d2) - S0 * Phi(-d1)
end

r, sigma, T = 0.05, 0.2, 1.0
Ns = collect(5:300)
err = [crr(100.0, 105.0, r, sigma, T, n) for n in Ns] .- bs(100.0, 105.0, r, sigma, T)

fig2 = Figure(size = (780, 340))
a0 = Axis(fig2[1, 1]; xlabel = L"\text{number of periods } N",
          title = L"\text{call: } N\,(C_N - C_{\mathrm{BS}}),\ S_0 = 100,\ K = 105")
hlines!(a0, [0.0]; color = PALETTE.ink, linewidth = 0.8)
for (par, col, lab) in ((0, PALETTE.blue, L"N \text{ even}"), (1, PALETTE.gold, L"N \text{ odd}"))
    sel = Ns .% 2 .== par
    scatterlines!(a0, Ns[sel], Ns[sel] .* err[sel]; color = col, markersize = 4, linewidth = 0.8, label = lab)
end
xlims!(a0, 0, 300); ylims!(a0, -2.6, 3.1)
axislegend(a0; position = :rt, orientation = :horizontal)

sig2, Kp, Nam = 0.3, 100.0, 500
s = collect(range(50, 150, length = 201))
am = [crr(x, Kp, r, sig2, T, Nam; kind = :put, american = true) for x in s]
eu = [bs(x, Kp, r, sig2, T; kind = :put) for x in s]
intr = max.(Kp .- s, 0.0)
s_star = maximum(s[am .- intr .< 1e-6])
a1 = Axis(fig2[1, 2]; xlabel = L"\text{spot price } S_0", title = L"\text{put value, } K = 100,\ \sigma = 0.3")
lines!(a1, s, intr; color = PALETTE.grey, linewidth = 1.5, label = L"\text{payoff } (K - S_0)^+")
lines!(a1, s, eu; color = PALETTE.blue, label = "European (Black–Scholes)")
lines!(a1, s, am; color = PALETTE.red, label = "American (CRR, N = 500)")
vlines!(a1, [s_star]; color = PALETTE.ink, linestyle = :dot)
text!(a1, s_star - 1.5, 8; text = @sprintf("S* ≈ %.1f", s_star), align = (:right, :center), fontsize = 12)
xlims!(a1, 50, 150); ylims!(a1, 0, 52)
axislegend(a1; position = :rt)
save_fig(fig2, @__DIR__, "convergence")
