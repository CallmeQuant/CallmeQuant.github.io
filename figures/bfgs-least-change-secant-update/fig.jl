# BFGS on the Rosenbrock function versus gradient descent and Newton's method, and the
# hereditary property of BFGS on a quadratic.
# Packages: CairoMakie, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra, Printf, Random

set_blog_theme!()

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
grad(x) = [-2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2), 200 * (x[2] - x[1]^2)]
hess(x) = [2 - 400 * x[2] + 1200 * x[1]^2  -400 * x[1]; -400 * x[1]  200.0]

function wolfe(x, p, g; c1 = 1e-4, c2 = 0.9, amax = 50.0)
    lo, hi, a = 0.0, Inf, 1.0
    f0, d0 = f(x), dot(g, p)
    for _ in 1:60
        if f(x + a * p) > f0 + c1 * a * d0
            hi = a
        elseif dot(grad(x + a * p), p) < c2 * d0
            lo = a
        else
            return a
        end
        a = isfinite(hi) ? (lo + hi) / 2 : min(2a, amax)
    end
    return a
end
function backtrack(x, p, g; c1 = 1e-4)
    a = 1.0
    while f(x + a * p) > f(x) + c1 * a * dot(g, p)
        a /= 2
    end
    return a
end
function optimize_path(method, x0, iters; tol = 1e-10)
    x = copy(x0); H = Matrix(1.0I, 2, 2); xs = [copy(x)]; gs = [norm(grad(x))]
    for _ in 1:iters
        g = grad(x)
        norm(g) < tol && break
        if method == :gd
            p = -g; a = 1e-3
        elseif method == :newton
            p = -(hess(x) \ g); a = backtrack(x, p, g)
        else
            p = -H * g; a = wolfe(x, p, g)
        end
        xn = x + a * p
        if method == :bfgs
            s = xn - x; y = grad(xn) - g; rho = 1 / dot(y, s)
            length(xs) == 1 && (H = dot(y, s) / dot(y, y) * Matrix(1.0I, 2, 2))
            V = I - rho * s * y'
            H = V * H * V' + rho * s * s'
        end
        x = xn; push!(xs, copy(x)); push!(gs, norm(grad(x)))
    end
    return reduce(hcat, xs)', gs
end

x0 = [-1.2, 1.0]
names_ = ["gradient descent", "BFGS", "Newton"]
runs = [optimize_path(:gd, x0, 3000), optimize_path(:bfgs, x0, 200), optimize_path(:newton, x0, 200)]
cols = [PALETTE.red, PALETTE.blue, PALETTE.green]

fig = Figure(size = (780, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"x_2", title = L"\text{(a) Rosenbrock function, start } (-1.2, 1)")
xg = range(-1.6, 1.5; length = 400); yg = range(-0.6, 1.6; length = 400)
F = [(1 - a)^2 + 100 * (c - a^2)^2 for a in xg, c in yg]
contour!(ax1, xg, yg, F; levels = exp.(range(log(0.05), log(800); length = 14)), color = PALETTE.grey, linewidth = 0.7)
for (nm, (xs, _), col) in zip(names_, runs, cols)
    lab = "$(nm) ($(size(xs, 1) - 1) it.)"
    if nm == "gradient descent"
        lines!(ax1, xs[:, 1], xs[:, 2]; color = col, linewidth = 1.2, label = lab)
    else
        scatterlines!(ax1, xs[:, 1], xs[:, 2]; color = col, linewidth = 1.5, markersize = 5, label = lab)
    end
end
scatter!(ax1, [1.0], [1.0]; marker = :star5, color = PALETTE.ink, markersize = 15)
scatter!(ax1, [x0[1]], [x0[2]]; marker = :rect, color = PALETTE.ink, markersize = 8)
scatter!(ax1, [runs[1][1][end, 1]], [runs[1][1][end, 2]]; color = PALETTE.red, markersize = 8)
xlims!(ax1, -1.6, 1.5); ylims!(ax1, -0.6, 1.6)
axislegend(ax1; position = :rb, labelsize = 11)

ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = L"\text{iteration } k", ylabel = L"\Vert\nabla f(x_k)\Vert",
           title = "(b) Gradient norm")
for ((_, gs), col) in zip(runs, cols)
    m = min(length(gs), 61)
    lines!(ax2, 0:m-1, max.(gs[1:m], 1e-16); color = col)   # Newton reaches an exact zero
end
text!(ax2, 58, 4e-3; text = "gradient descent:\n" * @sprintf("%.2f after 3000 it.", runs[1][2][end]),
      color = PALETTE.red, align = (:right, :top), fontsize = 12)
xlims!(ax2, 0, 60); ylims!(ax2, 1e-11, 1e4)
colsize!(fig.layout, 1, Relative(0.53))
save_fig(fig, @__DIR__, "rosenbrock")

# ---- Figure 2 ------------------------------------------------------------------------------
rng = MersenneTwister(8)
n = 8
Q = Matrix(qr(randn(rng, n, n)).Q)
A = Q * Diagonal(10 .^ range(0.3, 2.3; length = n)) * Q'
b = randn(rng, n); x = zeros(n); H = Matrix(1.0I, n, n)
eigs = [sort(real(eigvals(H * A)))]
for k in 1:n
    g = A * x - b; p = -H * g; a = -dot(g, p) / dot(p, A * p)
    s = a * p; y = A * s; rho = 1 / dot(y, s); V = I - rho * s * y'
    global H = V * H * V' + rho * s * s'
    global x = x + s
    push!(eigs, sort(real(eigvals(H * A))))
end
fig2 = Figure(size = (600, 330))
ax = Axis(fig2[1, 1]; yscale = log10, xlabel = L"\text{iteration } k", ylabel = L"\text{eigenvalues of } H_k A",
          xticks = 0:n, title = L"\text{Quadratic, } n = 8\text{, exact line search; top: number of eigenvalues equal to } 1",
          titlesize = 13)
hlines!(ax, [1.0]; color = PALETTE.blue, linestyle = :dot, linewidth = 1)
for k in 0:n
    e = eigs[k + 1]; one = abs.(e .- 1) .< 1e-6
    scatter!(ax, fill(k, count(.!one)), e[.!one]; color = PALETTE.grey, markersize = 9)
    scatter!(ax, fill(k, count(one)), e[one]; color = PALETTE.blue, markersize = 10)
    text!(ax, k, 400; text = string(count(one)), color = PALETTE.blue, align = (:center, :bottom), fontsize = 12)
end
ylims!(ax, 0.5, 900)
save_fig(fig2, @__DIR__, "hereditary-eigenvalues")
