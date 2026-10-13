# Coordinate ascent variational inference with a mean-field Gaussian family for a bivariate
# Gaussian target (note: variational inference and the evidence lower bound).
# Packages: CairoMakie, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra

set_blog_theme!()
const mu = [1.0, 0.5]
const m0 = [-1.5, 2.5]   # initial means; variances start at their optimal values 1 - rho^2

function cavi(rho, sweeps)
    Lam = inv([1.0 rho; rho 1.0])
    m = copy(m0); path = [copy(m)]
    for _ in 1:sweeps
        m[1] = mu[1] - Lam[1, 2] / Lam[1, 1] * (m[2] - mu[2]); push!(path, copy(m))
        m[2] = mu[2] - Lam[2, 1] / Lam[2, 2] * (m[1] - mu[1]); push!(path, copy(m))
    end
    return reduce(hcat, path)'      # rows are iterates
end
function kl_mf(m, rho)
    Lam = inv([1.0 rho; rho 1.0]); d = m .- mu
    return 0.5 * dot(d, Lam * d) - 0.5 * log(1 - rho^2)
end
function ellipse(S, r; n = 200)
    a = range(0, 2pi; length = n)
    L = cholesky(Symmetric(S)).L
    E = mu .+ r .* (L * vcat(cos.(a)', sin.(a)'))
    return E[1, :], E[2, :]
end

fig = Figure(size = (800, 370))
rho = 0.8
ax1 = Axis(fig[1, 1]; xlabel = L"z_1", ylabel = L"z_2", aspect = DataAspect(),
           title = L"\text{(a) } \rho = 0.8\text{: contours at radii 1 and 2}")
S = [1.0 rho; rho 1.0]
for r in (1, 2)
    lines!(ax1, ellipse(S, r)...; color = PALETTE.ink, linewidth = 1.6, label = r == 1 ? L"\text{target } p" : nothing)
    lines!(ax1, ellipse(Matrix((1 - rho^2) * I(2)), r)...; color = PALETTE.red, linewidth = 1.6,
           linestyle = :dash, label = r == 1 ? L"\text{optimal } q^\ast" : nothing)
end
P = cavi(rho, 8)
scatterlines!(ax1, P[:, 1], P[:, 2]; color = PALETTE.blue, markersize = 6, linewidth = 1.3, label = "CAVI means")
text!(ax1, m0[1] + 0.2, m0[2] + 0.15; text = "start", color = PALETTE.blue, fontsize = 12)
xlims!(ax1, -2.4, 3.6); ylims!(ax1, -2.2, 3.2)
axislegend(ax1; position = :rb, labelsize = 11)

ax2 = Axis(fig[1, 2]; xlabel = "sweep", yscale = log10, yticks = [0.1, 0.2, 0.5, 1, 2, 5],
           ylabel = L"\mathrm{KL}(q_t \,\Vert\, p) = \log p(x) - \mathrm{ELBO}(q_t)",
           title = L"\text{(b) gap after each coordinate update}")
for (k, r) in enumerate([0.5, 0.8, 0.95])
    Pr = cavi(r, 30)[2:end, :]
    kl = [kl_mf(Pr[i, :], r) for i in 1:size(Pr, 1)]
    lines!(ax2, (1:length(kl)) ./ 2, kl; color = CYCLE[k], label = L"\rho = %$r")
    hlines!(ax2, [-0.5 * log(1 - r^2)]; color = CYCLE[k], linewidth = 1, linestyle = :dot)
end
xlims!(ax2, 0, 15); ylims!(ax2, 0.1, 6)
axislegend(ax2; position = :rt)
colsize!(fig.layout, 2, Relative(0.53))
save_fig(fig, @__DIR__, "cavi-gaussian")
