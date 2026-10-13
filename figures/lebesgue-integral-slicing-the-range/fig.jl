# Riemann sums versus Lebesgue sums, the distribution function, and dyadic simple functions.
# Packages: CairoMakie, Statistics, Printf
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Statistics, Printf

set_blog_theme!()
f(x) = 1 + 0.5sin(2pi * x) + 0.3x * cos(7pi * x)
x = collect(range(0, 1; length = 20001)); fx = f.(x)
I = sum((fx[2:end] .+ fx[1:end-1]) ./ 2) * (x[2] - x[1])
G(t) = mean(fx .>= t)
sl = 1:5:20001

fig = Figure(size = (1040, 340))
ax1 = Axis(fig[1, 1]; xlabel = L"x", ylabel = L"f(x)")
n = 10; e = range(0, 1; length = n + 1)
R = 0.0
for i in 1:n
    h = f(e[i]); R += h * (e[i+1] - e[i])
    poly!(ax1, Rect(e[i], 0, e[i+1] - e[i], h); color = (PALETTE.blue, 0.25), strokecolor = PALETTE.blue, strokewidth = 0.7)
end
lines!(ax1, x[sl], fx[sl]; color = PALETTE.ink)
ax1.title = @sprintf("Riemann: slice the domain, sum = %.3f", R)

ax2 = Axis(fig[1, 2]; xlabel = L"x")
h = 0.25; levels = h:h:(maximum(fx) + h)
L = 0.0
for (k, y) in enumerate(levels)
    m = fx .>= y
    any(m) || continue
    L += h * mean(m)
    d = diff(vcat(0, Int.(m), 0)); s = findall(==(1), d); en = findall(==(-1), d) .- 1
    col = CYCLE[mod1(k, 5)]
    for (a, b) in zip(s, en)
        poly!(ax2, Rect(x[a], y - h, x[b] - x[a], h); color = (col, 0.3))
        lines!(ax2, [x[a], x[b]], [y, y]; color = col, linewidth = 3)
    end
end
lines!(ax2, x[sl], fx[sl]; color = PALETTE.ink)
ax2.title = @sprintf("Lebesgue: slice the range, sum = %.3f", L)

ax3 = Axis(fig[1, 3]; xlabel = L"\lambda(\{f \geq t\})", ylabel = L"t",
           title = @sprintf("Area = ∫ f dλ = %.3f", I))
tt = range(0, 1.75; length = 600); Gt = G.(tt)
band!(ax3, Point2f.(zeros(length(tt)), tt), Point2f.(Gt, tt); color = (PALETTE.grey, 0.3))
lines!(ax3, Gt, tt; color = PALETTE.ink)
for (k, y) in enumerate(levels)
    G(y) > 0 && lines!(ax3, [0, G(y)], [y, y]; color = CYCLE[mod1(k, 5)], linewidth = 3)
end
for ax in (ax1, ax2); xlims!(ax, 0, 1); ylims!(ax, 0, 1.75); end
xlims!(ax3, 0, 1.05); ylims!(ax3, 0, 1.75)
colsize!(fig.layout, 3, Relative(0.3))
save_fig(fig, @__DIR__, "riemann-vs-lebesgue")

fig = Figure(size = (1060, 290))
for nn in 1:4
    phi = min.(floor.(fx .* 2^nn) ./ 2^nn, nn)
    ax = Axis(fig[1, nn]; xlabel = L"x", xticks = [0, 0.5, 1], ylabel = nn == 1 ? L"\varphi_n(x)" : "", ygridvisible = false,
              title = @sprintf("n = %d: ∫φ = %.3f", nn, mean(phi)))
    hl = filter(<=(1.75), (1:nn*2^nn) ./ 2^nn)
    hlines!(ax, hl; color = (PALETTE.grey, 0.6), linewidth = 0.5)
    band!(ax, x[sl], zeros(length(sl)), phi[sl]; color = (PALETTE.blue, 0.25))
    lines!(ax, x[sl], phi[sl]; color = PALETTE.blue, linewidth = 1.2)
    lines!(ax, x[sl], fx[sl]; color = PALETTE.ink, linewidth = 1.6)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1.75)
    nn > 1 && hideydecorations!(ax; grid = false)
end
save_fig(fig, @__DIR__, "dyadic-approximations")
