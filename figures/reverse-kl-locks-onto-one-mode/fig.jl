# Gaussian fits to a two-component mixture under reverse and forward KL, and mean-field fits to a
# correlated Gaussian (note: zero-forcing in reverse Kullback-Leibler fits).
# Packages: CairoMakie, Distributions, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra

set_blog_theme!()
const ZG = collect(range(-10, 10; length = 2001))
const WZ = let w = pdf.(Normal(), ZG); w ./ sum(w) end      # quadrature weights for N(0, 1)

logaddexp(u, v) = max(u, v) + log1p(exp(-abs(u - v)))
log_p(x, a, w) = logaddexp(log(w) - (x - a)^2 / 2, log1p(-w) - (x + a)^2 / 2) - log(2π) / 2
reverse_kl(m, s, a, w) = sum(WZ .* (logpdf.(Normal(m, s), m .+ s .* ZG) .- log_p.(m .+ s .* ZG, a, w)))

const LS = collect(range(log(0.2), log(8.0); length = 600))
function reverse_profile(m, a, w)
    v = [reverse_kl(m, exp(l), a, w) for l in LS]
    k = argmin(v)
    return v[k], exp(LS[k])
end
function forward_profile(m, a, w)
    xx = collect(range(-a - 12, a + 12; length = 20001))
    lp = log_p.(xx, a, w)
    H = -sum(exp.(lp) .* lp) * step(range(-a - 12, a + 12; length = 20001))
    mu, v = (2w - 1) * a, 1 + 4w * (1 - w) * a^2
    return -H + log(2π * ℯ * (v + (m - mu)^2)) / 2
end

# ---------- Figure 1 ----------
a, w = 3.2, 0.6
mg = collect(range(-6, 6; length = 481))
rev = [reverse_profile(m, a, w) for m in mg]
rk = first.(rev)
fwd = [forward_profile(m, a, w) for m in mg]
loc = [i for i in 2:length(mg)-1 if rk[i] < rk[i-1] && rk[i] < rk[i+1]]
sort!(loc; by = i -> rk[i])
mu_p, sd_p = (2w - 1) * a, sqrt(1 + 4w * (1 - w) * a^2)
x = collect(range(-7.5, 7.5; length = 800))
pdens = exp.(log_p.(x, a, w))

fig = Figure(size = (780, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"x", ylabel = "density", title = "Target and Gaussian fits")
band!(ax1, x, zero(x), pdens; color = PALETTE.light)
lines!(ax1, x, pdens; color = PALETTE.ink, linewidth = 1.6, label = L"\text{target } p")
lines!(ax1, x, pdf.(Normal(mg[loc[1]], rev[loc[1]][2]), x); color = PALETTE.red,
       label = L"\arg\min \, \mathrm{KL}(q \,\Vert\, p)")
lines!(ax1, x, pdf.(Normal(mg[loc[2]], rev[loc[2]][2]), x); color = PALETTE.red, linestyle = :dash,
       linewidth = 1.6, label = "its other local minimum")
lines!(ax1, x, pdf.(Normal(mu_p, sd_p), x); color = PALETTE.blue, label = L"\arg\min \, \mathrm{KL}(p \,\Vert\, q)")
xlims!(ax1, -7.5, 7.5); ylims!(ax1, 0, 0.72)
axislegend(ax1; position = :lt, labelsize = 11)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{location } m \text{ of } q = \mathcal{N}(m, s^2)",
           ylabel = "divergence (profiled over s)", title = "Profiles in the location")
hlines!(ax2, [-log(w), -log(1 - w)]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
text!(ax2, -5.9, -log(w) - 0.04; text = L"-\log w", align = (:left, :top), fontsize = 12, color = PALETTE.grey)
text!(ax2, -5.9, -log(1 - w) - 0.04; text = L"-\log(1-w)", align = (:left, :top), fontsize = 12, color = PALETTE.grey)
lines!(ax2, mg, rk; color = PALETTE.red, label = L"\min_s \mathrm{KL}(\mathcal{N}(m,s^2) \,\Vert\, p)")
lines!(ax2, mg, fwd; color = PALETTE.blue, label = L"\min_s \mathrm{KL}(p \,\Vert\, \mathcal{N}(m,s^2))")
xlims!(ax2, -6, 6); ylims!(ax2, 0, 3)
axislegend(ax2; position = :ct, labelsize = 11)
save_fig(fig, @__DIR__, "mixture-fits")

# ---------- Figure 2 ----------
fig = Figure(size = (640, 340))
ax = Axis(fig[1, 1]; xlabel = L"\text{location } m", ylabel = L"\min_s \mathrm{KL}(\mathcal{N}(m,s^2) \,\Vert\, p_a)")
mg2 = collect(range(-5, 5; length = 201))
hlines!(ax, [log(2)]; color = PALETTE.grey, linestyle = :dot, linewidth = 1)
text!(ax, 4.95, log(2) + 0.03; text = L"\log 2", align = (:right, :bottom), fontsize = 12, color = PALETTE.grey)
for (k, aa) in enumerate([1.5, 2.5, 2.8, 3.5])
    lines!(ax, mg2, [reverse_profile(m, aa, 0.5)[1] for m in mg2]; color = CYCLE[k], label = L"a = %$(aa)")
end
xlims!(ax, -5, 5); ylims!(ax, 0, 1.8)
Legend(fig[0, 1], ax; orientation = :horizontal, tellwidth = false)
save_fig(fig, @__DIR__, "separation-profiles")

# ---------- Figure 3 ----------
function ellipse(S; r = 2.0, n = 300)
    t = range(0, 2π; length = n)
    L = cholesky(Symmetric(S)).L
    E = L * vcat(cos.(t)', sin.(t)') .* r
    return E[1, :], E[2, :]
end
fig = Figure(size = (700, 380))
axs = Axis[]
for (j, rho) in enumerate([0.6, 0.95])
    S = [1.0 rho; rho 1.0]
    Lam = inv(S)
    ax = Axis(fig[1, j]; title = L"\rho = %$(rho)", xlabel = L"x_1", ylabel = j == 1 ? L"x_2" : "",
              aspect = DataAspect())
    lines!(ax, ellipse(S)...; color = PALETTE.ink, linewidth = 1.6, label = L"\text{target } \mathcal{N}(0, \Sigma)")
    lines!(ax, ellipse(Diagonal(1 ./ diag(Lam)) |> Matrix)...; color = PALETTE.red,
           label = L"\text{reverse KL: } v_i = 1/\Lambda_{ii}")
    lines!(ax, ellipse(Diagonal(diag(S)) |> Matrix)...; color = PALETTE.blue, linestyle = :dash,
           label = L"\text{forward KL: } v_i = \Sigma_{ii}")
    xlims!(ax, -2.6, 2.6); ylims!(ax, -2.6, 2.6)
    push!(axs, ax)
end
hideydecorations!(axs[2]; grid = false)
Legend(fig[2, 1:2], axs[1]; orientation = :horizontal, tellheight = true)
save_fig(fig, @__DIR__, "mean-field-ellipses")
