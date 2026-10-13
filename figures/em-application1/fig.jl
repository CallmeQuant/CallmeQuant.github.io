# EM for a three-component bivariate Gaussian mixture.
# gmm-data: complete data coloured by component versus the observed data.
# em-iterations: fitted components at several EM iterations and the log-likelihood trace.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random, Statistics

set_blog_theme!()
rng = MersenneTwister(91)
w_true = [0.4, 0.3, 0.3]
mu_true = [[-3.0, 0.0], [2.0, 2.5], [2.5, -2.0]]
cov_true = [[1.2 0.6; 0.6 1.0], [0.8 -0.3; -0.3 0.6], [1.5 0.0; 0.0 0.4]]
n = 450
z = rand(rng, Categorical(w_true), n)
X = permutedims(hcat([rand(rng, MvNormal(mu_true[k], cov_true[k])) for k in z]...))
cols = [PALETTE.blue, PALETTE.red, PALETTE.green]

fig = Figure(size = (720, 340))
ax1 = Axis(fig[1, 1]; title = L"\text{(a) complete data } (x_i, z_i)", xlabel = L"x_{i1}", ylabel = L"x_{i2}",
           aspect = DataAspect())
for k in 1:3
    scatter!(ax1, X[z .== k, 1], X[z .== k, 2]; color = (cols[k], 0.75), markersize = 5, label = L"z_i = %$k")
end
axislegend(ax1; position = :lt)
ax2 = Axis(fig[1, 2]; title = L"\text{(b) observed data } x_i", xlabel = L"x_{i1}", aspect = DataAspect())
scatter!(ax2, X[:, 1], X[:, 2]; color = (PALETTE.grey, 0.75), markersize = 5)
linkaxes!(ax1, ax2)
save_fig(fig, @__DIR__, "gmm-data")

logsumexp_rows(A) = (m = maximum(A; dims = 2); vec(m .+ log.(sum(exp.(A .- m); dims = 2))))

function e_step(w, mu, S)
    logp = hcat([log(w[k]) .+ logpdf(MvNormal(mu[k], Symmetric(S[k])), X') for k in 1:3]...)
    ll = logsumexp_rows(logp)
    return exp.(logp .- ll), sum(ll)
end

function m_step(g)
    Nk = vec(sum(g; dims = 1))
    mu = [vec(sum(g[:, k] .* X; dims = 1)) ./ Nk[k] for k in 1:3]
    S = [begin D = X .- mu[k]'; (D .* g[:, k])' * D ./ Nk[k] end for k in 1:3]
    return Nk ./ n, mu, S
end

function ellipse(mu, S; level = 5.991)
    F = eigen(Symmetric(S)); th = range(0, 2pi; length = 200)
    pts = hcat(cos.(th), sin.(th)) * Diagonal(sqrt.(level .* F.values)) * F.vectors'
    return pts[:, 1] .+ mu[1], pts[:, 2] .+ mu[2]
end

init = [findall(z .== 1)[1], findall(z .== 1)[2], findall(z .== 3)[1]]
par = (fill(1 / 3, 3), [X[i, :] for i in init], [cov(X) for _ in 1:3])
snaps = Dict(0 => par); lls = Float64[]
T = 0
for it in 1:200
    g, ll = e_step(par...)
    push!(lls, ll)
    global par = m_step(g)
    snaps[it] = par
    global T = it
    (it > 1 && lls[end] - lls[end-1] < 1e-8) && break
end
shown = [0, 1, 4, 10, T]
# rows: RGB of the palette blue, red and green (responsibility-weighted colour mix)
rgbm = [0x2b 0x4c 0x9b; 0xb5 0x48 0x2f; 0x2e 0x7d 0x5b] ./ 255

fig = Figure(size = (800, 540))
for (j, t) in enumerate(shown)
    ax = Axis(fig[(j - 1) ÷ 3 + 1, (j - 1) % 3 + 1]; title = "iteration $t", aspect = DataAspect(),
              xticklabelsize = 11, yticklabelsize = 11)
    w, mu, S = snaps[t]
    g, _ = e_step(w, mu, S)
    mix = g * rgbm
    scatter!(ax, X[:, 1], X[:, 2]; color = [RGBAf(mix[i, 1], mix[i, 2], mix[i, 3], 0.6) for i in 1:n],
             markersize = 4)
    for k in 1:3
        ex, ey = ellipse(mu[k], S[k])
        lines!(ax, ex, ey; color = PALETTE.ink, linewidth = 1.3)
        scatter!(ax, [mu[k][1]], [mu[k][2]]; marker = :xcross, markersize = 10, color = PALETTE.ink)
    end
    xlims!(ax, -7, 7); ylims!(ax, -6, 6)
end
axl = Axis(fig[2, 3]; title = L"\text{log-likelihood } \ell(\Theta_t)", xlabel = L"\text{iteration } t",
           xticklabelsize = 11, yticklabelsize = 11)
scatterlines!(axl, 0:length(lls)-1, lls; color = PALETTE.blue, markersize = 5)
save_fig(fig, @__DIR__, "em-iterations")
