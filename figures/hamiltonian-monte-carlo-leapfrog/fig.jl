# Leapfrog trajectories, energy error and integration time for HMC on Gaussian targets.
# Packages: CairoMakie, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random, Statistics, Printf

set_blog_theme!()

# Target N(0, Sigma), U(q) = q' Sigma^{-1} q / 2, identity mass matrix.
Sigma = [1.0 0.95; 0.95 1.0]
Prec = inv(Sigma)
sig_min = sqrt(minimum(eigvals(Symmetric(Sigma))))
Hfun(q, p) = 0.5 * dot(q, Prec * q) + 0.5 * dot(p, p)

function leapfrog(q, p, eps, L)
    qs, ps = [copy(q)], [copy(p)]
    for _ in 1:L
        p = p .- 0.5 * eps .* (Prec * q)
        q = q .+ eps .* p
        p = p .- 0.5 * eps .* (Prec * q)
        push!(qs, q); push!(ps, p)
    end
    return qs, ps
end

# ---- Figure 1
q0, p0 = [-1.2, -0.9], [1.4, -0.6]
cases = [(0.05, 64, PALETTE.blue), (0.40, 8, PALETTE.green), (0.46, 3, PALETTE.red)]
g = range(-2.6, 2.6; length = 300)
D = [exp(-Hfun([x, y], [0.0, 0.0])) for x in g, y in g]

fig = Figure(size = (780, 360))
ax1 = Axis(fig[1, 1]; title = "Leapfrog paths in position space", xlabel = L"q_1", ylabel = L"q_2",
           aspect = DataAspect())
contour!(ax1, g, g, D; levels = [0.05, 0.25, 0.5, 0.8], color = PALETTE.grey, linewidth = 1)
for (eps, L, col) in cases
    qs, _ = leapfrog(q0, p0, eps, L)
    xs, ys = first.(qs), last.(qs)
    lines!(ax1, xs, ys; color = col, linewidth = 1.4)
    scatter!(ax1, xs, ys; color = col, markersize = L > 20 ? 5 : 7, label = L"\varepsilon = %$eps")
end
scatter!(ax1, [q0[1]], [q0[2]]; marker = :rect, color = PALETTE.ink, markersize = 10)
xlims!(ax1, -2.6, 2.6); ylims!(ax1, -2.6, 2.6)
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{integration time } k\varepsilon",
           ylabel = L"|H(z_k) - H(z_0)|",
           title = L"\text{Energy error (stability limit } 2\sigma_{\min} = %$(round(2sig_min; digits = 3)))")
for (eps, _, col) in cases
    L = round(Int, 6 / eps)
    qs, ps = leapfrog(q0, p0, eps, L)
    err = [abs(Hfun(q, p) - Hfun(q0, p0)) for (q, p) in zip(qs, ps)]
    t = eps .* (0:L)
    lines!(ax2, t[2:end], err[2:end]; color = col, linewidth = 1.6)
end
xlims!(ax2, 0, 6); ylims!(ax2, 1e-5, 1e4)
colsize!(fig.layout, 2, Relative(0.55))
save_fig(fig, @__DIR__, "leapfrog-energy")

# ---- Figure 2: lag-one autocorrelation from one HMC transition started in stationarity
rng = MersenneTwister(11)
N = 20_000
function one_step(C, eps, T, rng)
    d = size(C, 1); P = inv(C * C')
    q0 = randn(rng, N, d) * C'; p = randn(rng, N, d)
    L = max(1, round(Int, T / eps)); eps = T / L; q = copy(q0)
    energy(q, p) = 0.5 .* sum((q * P) .* q; dims = 2)[:] .+ 0.5 .* sum(p .^ 2; dims = 2)[:]
    h0 = energy(q, p)
    for _ in 1:L
        p = p .- 0.5 * eps .* (q * P); q = q .+ eps .* p; p = p .- 0.5 * eps .* (q * P)
    end
    acc = log.(rand(rng, N)) .< h0 .- energy(q, p)
    q[.!acc, :] = q0[.!acc, :]
    return q0, q
end

fig = Figure(size = (780, 340))
T = range(0, 2pi; length = 400); Te = range(0.15, 2pi; length = 24)
ax = Axis(fig[1, 1]; title = L"N(0, 1)", xlabel = L"\text{integration time } T",
          ylabel = "lag-one autocorrelation",
          xticks = ([0, pi / 2, pi, 3pi / 2, 2pi], [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"]))
emp = [one_step(ones(1, 1), 0.05, t, rng) for t in Te]
lines!(ax, T, cos.(T); color = PALETTE.blue, label = L"h(q) = q:\ \cos T")
lines!(ax, T, cos.(T) .^ 2; color = PALETTE.red, label = L"h(q) = q^2:\ \cos^2 T")
scatter!(ax, Te, [cor(a[:, 1], b[:, 1]) for (a, b) in emp]; color = PALETTE.blue, markersize = 7)
scatter!(ax, Te, [cor(a[:, 1] .^ 2, b[:, 1] .^ 2) for (a, b) in emp]; color = PALETTE.red, markersize = 7)
hlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.8)
ylims!(ax, -1.15, 1.6)
axislegend(ax; position = :ct)

E = eigen(Symmetric(Sigma)); sd = sqrt.(E.values); V = E.vectors     # ascending: column 2 is wide
C = cholesky(Symmetric(Sigma)).L
T = range(0, 2pi * sd[2]; length = 600); Te = range(0.1, 2pi * sd[2]; length = 26)
emp = [one_step(Matrix(C), 0.1 * sd[1], t, rng) for t in Te]
ax = Axis(fig[1, 2]; title = L"N(0, \Sigma),\ \rho = 0.95\text{: principal components}",
          xlabel = L"\text{integration time } T")
for (j, col, name) in [(2, PALETTE.blue, "wide"), (1, PALETTE.red, "narrow")]
    lines!(ax, T, cos.(T ./ sd[j]); color = col, linewidth = 1.6,
           label = @sprintf("%s axis, σ = %.2f", name, sd[j]))
    scatter!(ax, Te, [cor(a * V[:, j], b * V[:, j]) for (a, b) in emp]; color = col, markersize = 7)
end
hlines!(ax, [0.0]; color = PALETTE.ink, linewidth = 0.8)
ylims!(ax, -1.15, 1.6)
axislegend(ax; position = :ct)
save_fig(fig, @__DIR__, "integration-time")
