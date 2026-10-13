# Union bound for the maximum of m = 20 equicorrelated standard normals (note: Boole's inequality).
# Packages: CairoMakie, Distributions, QuadGK
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, QuadGK

set_blog_theme!()
m, alpha = 20, 0.05
t = collect(range(0.0, 4.5; length = 451))
N01 = Normal()
tail = ccdf.(N01, t)

# P(max Z_i > t) for Z_i = sqrt(rho) W + sqrt(1 - rho) E_i
function p_max_exceeds(tt, rho)
    rho == 1 && return ccdf(N01, tt)
    rho == 0 && return -expm1(m * logcdf(N01, tt))
    f(w) = exp(m * logcdf(N01, (tt - sqrt(rho) * w) / sqrt(1 - rho))) * pdf(N01, w)
    return 1 - quadgk(f, -Inf, Inf; rtol = 1e-10)[1]
end

union = m .* tail
bonf2 = m .* tail .- m * (m - 1) / 2 .* tail .^ 2
t_bonf = quantile(N01, 1 - alpha / m)

fig = Figure(size = (800, 330))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{threshold } t", yscale = log10,
           title = L"\mathrm{P}(\max_i Z_i > t),\ m = 20")
vlines!(ax1, [t_bonf]; color = PALETTE.violet, linewidth = 1, linestyle = :dot)
lines!(ax1, t, min.(union, 1); color = PALETTE.ink, linewidth = 1.6, linestyle = :dash,
       label = L"\text{union bound } m\,\bar\Phi(t)")
lines!(ax1, t, p_max_exceeds.(t, 0.0); color = CYCLE[1], label = L"\text{exact}, \rho = 0")
lines!(ax1, t, p_max_exceeds.(t, 1.0); color = CYCLE[4], label = L"\text{exact}, \rho = 1")
pos = bonf2 .> 0
lines!(ax1, t[pos], bonf2[pos]; color = PALETTE.grey, linewidth = 1.6, linestyle = :dot,
       label = L"\text{Bonferroni lower bound}, \rho = 0")
text!(ax1, t_bonf + 0.1, 0.4; text = L"t = \bar\Phi^{-1}(\alpha/m)", color = PALETTE.violet,
      fontsize = 13, align = (:left, :center))
xlims!(ax1, 0, 4.5); ylims!(ax1, 1e-4, 1.6)
axislegend(ax1; position = :lb, labelsize = 11)

ax2 = Axis(fig[1, 2]; xlabel = L"\text{threshold } t", yscale = log10,
           title = "union bound / exact probability", yticks = [1, 2, 5, 10, 20])
hlines!(ax2, [1.0]; color = PALETTE.grey, linewidth = 1, linestyle = :dash)
lines!(ax2, [t_bonf, t_bonf], [0.8, 28]; color = PALETTE.violet, linewidth = 1, linestyle = :dot)
for (k, rho) in enumerate([0.0, 0.5, 0.9, 1.0])
    lines!(ax2, t, union ./ p_max_exceeds.(t, rho); color = CYCLE[k], label = L"\rho = %$rho")
end
xlims!(ax2, 0, 4.5); ylims!(ax2, 0.8, 70)
axislegend(ax2; position = :ct, orientation = :horizontal)
save_fig(fig, @__DIR__, "union-bound-maximum")
