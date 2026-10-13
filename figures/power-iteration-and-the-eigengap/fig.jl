# Power iteration on symmetric 200 x 200 matrices: angle error and Rayleigh quotient error.
# Packages: CairoMakie, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random

set_blog_theme!()
rng = MersenneTwister(11)
n, K, FLOOR = 200, 150, 1e-15
V = Matrix(qr(randn(rng, n, n)).Q)
x0 = randn(rng, n)

function make(r)
    lam = vcat(1.0, r, (rand(rng, n - 2) .- 0.5) .* r)      # uniform on [-r/2, r/2]
    return V * Diagonal(lam) * V', lam
end
function power_run(A, lam)
    v1 = V[:, 1]; x = x0 / norm(x0)
    tn = zeros(K + 1); rq = zeros(K + 1)
    for k in 0:K
        c = dot(v1, x)
        tn[k + 1] = norm(x - c * v1) / abs(c)
        rq[k + 1] = abs(dot(x, A * x) - lam[1])
        y = A * x; x = y / norm(y)
    end
    return tn, rq
end
function cut(v)
    v = copy(v); i = findfirst(<(FLOOR), v)
    i === nothing || (v[i+1:end] .= NaN)
    return v
end

ratios = [0.5, 0.8, 0.9, 0.95, 0.99]
ks = 0:K
fig = Figure(size = (760, 340))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = L"\text{iteration } k", ylabel = L"\tan\theta_k",
           title = L"\text{(a) Angle to } v_1 \text{ for } \lambda_2/\lambda_1 \text{ as labelled}")
for (i, r) in enumerate(ratios)
    A, lam = make(r)
    tn = cut(power_run(A, lam)[1])
    lines!(ax1, ks, tn; color = CYCLE[i])
    last = findlast(isfinite, tn)
    text!(ax1, ks[last] + 3, max(tn[last], 3e-16); text = string(r), color = CYCLE[i],
          align = (:left, :center), fontsize = 12)
end
xlims!(ax1, 0, 172); ylims!(ax1, 1e-16, 1e3)

ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{iteration } k",
           title = L"\text{(b) } \lambda_2/\lambda_1 = 0.9\text{: eigenvalue error is squared}")
A, lam = make(0.9)
tn, rq = power_run(A, lam)
lines!(ax2, ks, cut(tn ./ sqrt.(1 .+ tn .^ 2)); color = PALETTE.blue, label = L"\sin\theta_k")
lines!(ax2, ks, cut(rq); color = PALETTE.red, label = L"\vert\rho(x_k) - \lambda_1\vert")
ylims!(ax2, 1e-16, 3)
axislegend(ax2; position = :lb)
save_fig(fig, @__DIR__, "power-iteration-rates")
