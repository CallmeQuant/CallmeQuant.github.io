# Kalman filter for a constant-velocity model with noisy position measurements.
# (a) true position, observations and filtered mean with 95% bands;
# (b) predicted and filtered variance of the position for the first 25 steps.
# Packages: CairoMakie, Distributions, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, LinearAlgebra, Random

set_blog_theme!()
rng = MersenneTwister(721)
T, dt, q, r = 60, 1.0, 0.02, 4.0
F = [1.0 dt; 0.0 1.0]
Q = q .* [dt^3/3 dt^2/2; dt^2/2 dt]
H = [1.0 0.0]
m0, P0 = [0.0, 0.0], Diagonal([4.0, 0.25])

z = zeros(T + 1, 2); y = zeros(T)
z[1, :] = rand(rng, MvNormal(m0, Matrix(P0)))
for t in 2:T+1
    z[t, :] = F * z[t-1, :] + rand(rng, MvNormal(zeros(2), Q))
    y[t-1] = z[t, 1] + sqrt(r) * randn(rng)
end

m, P = copy(m0), Matrix(P0)
mf = zeros(T); vf = zeros(T); vp = zeros(T)
for t in 1:T
    mp = F * m; Pp = F * P * F' + Q
    S = (H * Pp * H')[1, 1] + r
    K = Pp * H' ./ S
    global m = mp + vec(K) .* (y[t] - (H * mp)[1])
    global P = Pp - K * S * K'
    mf[t] = m[1]; vf[t] = P[1, 1]; vp[t] = Pp[1, 1]
end

ts = 1:T
fig = Figure(size = (780, 400))
ax1 = Axis(fig[1, 1]; xlabel = L"\text{time } t", ylabel = "position", title = "(a) filtering the position")
band!(ax1, ts, mf .- 1.96 .* sqrt.(vf), mf .+ 1.96 .* sqrt.(vf); color = (PALETTE.blue, 0.18), label = "95% band")
scatter!(ax1, ts, y; color = PALETTE.grey, markersize = 6, label = L"\text{observation } y_t")
lines!(ax1, ts, z[2:end, 1]; color = PALETTE.ink, linestyle = :dash, linewidth = 1.3, label = "true position")
lines!(ax1, ts, mf; color = PALETTE.blue, label = L"\text{filtered mean } (\mu_{t\mid t})_1")
Legend(fig[2, 1], ax1; orientation = :horizontal, nbanks = 2)

k = 25
ax2 = Axis(fig[1, 2]; xlabel = L"\text{time } t", title = "(b) position variance")
tt = repeat(1:k, inner = 2)
vv = vec(permutedims(hcat(vp[1:k], vf[1:k])))
lines!(ax2, tt, vv; color = PALETTE.grey, linewidth = 1)
scatter!(ax2, 1:k, vp[1:k]; color = PALETTE.red, markersize = 8, label = L"\text{predicted } (\Sigma_{t\mid t-1})_{11}")
scatter!(ax2, 1:k, vf[1:k]; color = PALETTE.blue, markersize = 8, label = L"\text{filtered } (\Sigma_{t\mid t})_{11}")
ylims!(ax2, 0, nothing)
Legend(fig[2, 2], ax2; orientation = :vertical)
colsize!(fig.layout, 1, Relative(0.62))
save_fig(fig, @__DIR__, "tracking")
