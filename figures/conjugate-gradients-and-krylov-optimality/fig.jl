# Conjugate gradients versus steepest descent: 2-d paths and A-norm error for n = 400.
# Packages: CairoMakie, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Random

set_blog_theme!()

function steepest(A, b, x0, iters)
    x = copy(x0); xs = [copy(x)]
    for _ in 1:iters
        r = b - A * x
        dot(r, r) == 0 && break
        x = x + (dot(r, r) / dot(r, A * r)) * r
        push!(xs, copy(x))
    end
    return reduce(hcat, xs)'          # iterations x n
end
function cg(A, b, x0, iters)
    x = copy(x0); xs = [copy(x)]
    r = b - A * x; p = copy(r); rr = dot(r, r)
    for _ in 1:iters
        rr == 0 && break
        Ap = A * p; a = rr / dot(p, Ap)
        x = x + a * p; r = r - a * Ap
        rr_new = dot(r, r); p = r + (rr_new / rr) * p; rr = rr_new
        push!(xs, copy(x))
    end
    return reduce(hcat, xs)'
end

# exact-arithmetic CG: minimize the A-norm error over x0 + K_k with a fully reorthogonalized Krylov basis
function krylov_optimum(A, b, x0, iters)
    r0 = b - A * x0; V = reshape(r0 / norm(r0), :, 1); xs = [copy(x0)]
    for _ in 1:iters
        y = (V' * A * V) \ (V' * r0); push!(xs, x0 + V * y)
        w = A * V[:, end]
        for _ in 1:2
            w = w - V * (V' * w)
        end
        V = hcat(V, w / norm(w))
    end
    return reduce(hcat, xs)'
end

th = deg2rad(30)
R = [cos(th) -sin(th); sin(th) cos(th)]
A2 = R * Diagonal([1.0, 10.0]) * R'
x0 = R * [-2.7, 0.27]                      # worst-case start for steepest descent
sd = steepest(A2, zeros(2), x0, 25); cgp = cg(A2, zeros(2), x0, 2)

rng = MersenneTwister(5)
n, K, kappa = 400, 160, 1000.0
Q = Matrix(qr(randn(rng, n, n)).Q)
lam_spread = 10 .^ range(0, log10(kappa); length = n)
lam_clust = vcat(1 .+ rand(rng, n - 5), [200.0, 400, 600, 800, 1000])
xs_true = randn(rng, n)
anorm_err(A, X) = [sqrt(dot(X[i, :] - xs_true, A * (X[i, :] - xs_true))) for i in 1:size(X, 1)]
res = Dict{Tuple{String, String}, Vector{Float64}}()
for (name, lam) in (("spread", lam_spread), ("clustered", lam_clust))
    A = Q * Diagonal(lam) * Q'; b = A * xs_true
    for (m, f) in (("cg", cg), ("sd", steepest))
        e = anorm_err(A, f(A, b, zeros(n), K))
        res[(name, m)] = e ./ e[1]
    end
    if name == "clustered"
        e = anorm_err(A, krylov_optimum(A, b, zeros(n), 30))
        res[(name, "exact")] = e ./ e[1]
    end
end
trim(v) = (v = copy(v); v[v .< 1e-11] .= NaN; v)

fig = Figure(size = (780, 350))
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = L"x_1", ylabel = L"x_2",
           title = L"\text{(a) } n = 2,\ \kappa = 10")
xg = range(-3, 3; length = 250); yg = range(-2.2, 2.2; length = 250)
F = [0.5 * dot([a, c], A2 * [a, c]) for a in xg, c in yg]
contour!(ax1, xg, yg, F; levels = exp.(range(log(0.02), log(30); length = 10)), color = PALETTE.grey, linewidth = 0.8)
scatterlines!(ax1, cgp[:, 1], cgp[:, 2]; color = (PALETTE.blue, 0.55), linewidth = 4, markersize = 9, label = "conjugate gradients")
scatterlines!(ax1, sd[:, 1], sd[:, 2]; color = PALETTE.red, linewidth = 1.2, markersize = 5, label = "steepest descent")
scatter!(ax1, [0.0], [0.0]; marker = :star5, color = PALETTE.ink, markersize = 14)
xlims!(ax1, -3, 3); ylims!(ax1, -2.2, 2.2)
axislegend(ax1; position = :rb, labelsize = 11)

ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{iteration } k",
           ylabel = L"\Vert x_k - x_\star\Vert_A / \Vert x_0 - x_\star\Vert_A",
           title = L"\text{(b) } n = 400,\ \kappa = 1000")
ks = 0:K
lines!(ax2, 0:length(res[("spread", "sd")]) - 1, trim(res[("spread", "sd")]); color = PALETTE.red, label = "steepest descent, spread")
lines!(ax2, 0:length(res[("spread", "cg")]) - 1, trim(res[("spread", "cg")]); color = PALETTE.blue, label = "CG, spread spectrum")
lines!(ax2, 0:length(res[("clustered", "cg")]) - 1, trim(res[("clustered", "cg")]); color = PALETTE.green, label = "CG, clustered + 5 outliers")
lines!(ax2, 0:length(res[("clustered", "exact")]) - 1, trim(res[("clustered", "exact")]); color = PALETTE.green,
       linestyle = :dash, label = "same, exact arithmetic")
lines!(ax2, ks, 2 .* ((sqrt(kappa) - 1) / (sqrt(kappa) + 1)) .^ ks; color = PALETTE.blue, linestyle = :dot,
       label = L"2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k")
ylims!(ax2, 1e-10, 3)
axislegend(ax2; position = :rb, labelsize = 11)
colsize!(fig.layout, 1, Relative(0.44))
save_fig(fig, @__DIR__, "cg-vs-steepest-descent")
