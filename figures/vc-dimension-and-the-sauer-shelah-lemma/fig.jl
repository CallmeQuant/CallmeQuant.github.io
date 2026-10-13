# VC dimension. Figure 1: three points shattered by half-planes; Radon partitions of four points.
# Figure 2: growth functions against 2^n and the Sauer-Shelah bound.
# Packages: CairoMakie, LinearAlgebra
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using LinearAlgebra

set_blog_theme!()
POS, NEG = PALETTE.blue, PALETTE.red
P3 = [0.0 0.0; 1.0 0.15; 0.35 1.0]

function sep_line(P, y)
    (all(y) || !any(y)) && return nothing
    odd = sum(y) == 1 ? y : .!y
    s = vec(P[odd, :]); others = P[.!odd, :]
    m = 0.5 .* (s .+ vec(sum(others; dims = 1)) ./ 2)
    d = others[2, :] .- others[1, :]
    w = [-d[2], d[1]]
    dot(w, s .- m) < 0 && (w = -w)
    sum(y) != 1 && (w = -w)
    return w, -dot(w, m)
end

function frame!(ax)
    hidedecorations!(ax); ax.leftspinecolor = PALETTE.light; ax.bottomspinecolor = PALETTE.light
    ax.topspinevisible = true; ax.rightspinevisible = true
    ax.topspinecolor = PALETTE.light; ax.rightspinecolor = PALETTE.light
end

fig = Figure(size = (780, 500))
Label(fig[1, 1:4], "Three points: all 8 labelings are realized by half-planes (shaded side = label 1)";
      halign = :left, fontsize = 14)
labelings = [Bool[a, b, c] for a in (false, true) for b in (false, true) for c in (false, true)]
for (k, y) in enumerate(labelings)
    r, c = divrem(k - 1, 4)
    ax = Axis(fig[2 + r, c + 1]; aspect = DataAspect())
    sl = sep_line(P3, y)
    if sl !== nothing
        w, b = sl; xs = [-0.6, 1.6]; ys = -(w[1] .* xs .+ b) ./ w[2]
        top = w[2] > 0 ? 10.0 : -10.0
        poly!(ax, Point2f[(xs[1], ys[1]), (xs[2], ys[2]), (xs[2], top), (xs[1], top)]; color = (POS, 0.10))
        lines!(ax, xs, ys; color = PALETTE.ink, linewidth = 1.2)
    elseif all(y)
        poly!(ax, Point2f[(-0.35, -0.3), (1.35, -0.3), (1.35, 1.3), (-0.35, 1.3)]; color = (POS, 0.10))
    end
    scatter!(ax, P3[y, 1], P3[y, 2]; color = POS, markersize = 9)
    scatter!(ax, P3[.!y, 1], P3[.!y, 2]; color = :white, strokecolor = NEG, strokewidth = 1.5, markersize = 9)
    xlims!(ax, -0.35, 1.35); ylims!(ax, -0.3, 1.3); frame!(ax)
end
Label(fig[4, 1:4], "Four points: the Radon partition (filled vs open) is never realized, because the convex hulls meet";
      halign = :left, fontsize = 14)
cases = [([0 0; 1.2 0.1; 1.1 1.0; 0.1 0.9], [1, 3], "convex position: the diagonals cross"),
         ([0 0; 1.3 0.1; 0.6 1.1; 0.62 0.42], [4], "one point inside the others' triangle")]
for (c, (P, I, ttl)) in enumerate(cases)
    ax = Axis(fig[5, 2c-1:2c]; title = ttl, titlesize = 13)
    J = setdiff(1:4, I)
    for (S, col) in [(I, POS), (J, NEG)]
        Q = P[S, :]
        size(Q, 1) == 2 && lines!(ax, Q[:, 1], Q[:, 2]; color = col, linewidth = 1.6)
        size(Q, 1) >= 3 && poly!(ax, Point2f.(Q[:, 1], Q[:, 2]); color = (col, 0.12), strokecolor = col, strokewidth = 1.2)
    end
    scatter!(ax, P[I, 1], P[I, 2]; color = POS, markersize = 11)
    scatter!(ax, P[J, 1], P[J, 2]; color = :white, strokecolor = NEG, strokewidth = 1.5, markersize = 11)
    xlims!(ax, -0.3, 1.6); ylims!(ax, -0.25, 1.3); frame!(ax)
end
rowsize!(fig.layout, 5, Relative(0.42))
save_fig(fig, @__DIR__, "shattering")

n = collect(1:30)
fig2 = Figure(size = (640, 350))
ax = Axis(fig2[1, 1]; yscale = log10, xlabel = L"n", ylabel = L"\text{growth function } \Pi_{\mathcal{H}}(n)")
lines!(ax, n, 2.0 .^ n; color = PALETTE.ink, linestyle = :dash, label = L"2^n")
for (d, f, col, lab) in [(1, n .+ 1, PALETTE.blue, L"\text{thresholds } (d = 1)\text{: } n + 1"),
                         (2, binomial.(n, 2) .+ n .+ 1, PALETTE.green, L"\text{intervals } (d = 2)\text{: } \binom{n}{2} + n + 1"),
                         (3, n .^ 2 .- n .+ 2, PALETTE.red, L"\text{half-planes in } \mathbb{R}^2 \ (d = 3)\text{: } n^2 - n + 2")]
    scatter!(ax, n, Float64.(f); color = col, markersize = 6, label = lab)
    lines!(ax, n, [Float64(sum(binomial(m, i) for i in 0:d)) for m in n]; color = col, linewidth = 1)
end
xlims!(ax, 0, 31); ylims!(ax, 1, 1e5)
axislegend(ax; position = :lt, labelsize = 12)
text!(ax, 30.5, 2.0; text = L"\text{lines: Sauer–Shelah bound } \sum_{i \leq d}\binom{n}{i}", align = (:right, :bottom), fontsize = 13)
save_fig(fig2, @__DIR__, "growth-functions")
