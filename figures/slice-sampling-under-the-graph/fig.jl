# Slice sampling with stepping out and shrinkage on a two-component normal mixture.
# Packages: CairoMakie, Distributions, Random, Statistics
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Distributions, Random, Statistics, Printf

set_blog_theme!()

# Target: 0.55 N(-2.5, 0.5^2) + 0.45 N(2.5, 0.7^2).
f(x) = 0.55 * pdf(Normal(-2.5, 0.5), x) + 0.45 * pdf(Normal(2.5, 0.7), x)

function slice_step(x, w, rng)
    u = rand(rng) * f(x)
    L = x - w * rand(rng); R = L + w
    steps = [(L, R)]
    while f(L) > u
        L -= w; push!(steps, (L, R))
    end
    while f(R) > u
        R += w; push!(steps, (L, R))
    end
    rejected = Float64[]
    y = x
    while true
        y = L + (R - L) * rand(rng)
        f(y) > u && break
        push!(rejected, y)
        y < x ? (L = y) : (R = y)
    end
    return y, u, steps, rejected
end

function run(x0, w, T, rng)
    out = zeros(T); x = x0
    for t in 1:T
        x = slice_step(x, w, rng)[1]; out[t] = x
    end
    return out
end

# ---- Figure 1: anatomy of one update
xs = range(-5, 5; length = 1000)
x0 = -2.05
rng = MersenneTwister(3)
function illustrative_step(x0, rng)   # a step whose level cuts both modes
    while true
        res = slice_step(x0, 1.0, rng)
        (0.08 < res[2] < 0.16 && length(res[4]) >= 1) && return res
    end
end
y, u, steps, rej = illustrative_step(x0, rng)

fig = Figure(size = (740, 340))
ax = Axis(fig[1, 1]; xlabel = L"x", ylabel = L"f(x)", yticks = [0, 0.2, 0.4])
band!(ax, xs, zeros(length(xs)), f.(xs); color = PALETTE.light)
lines!(ax, xs, f.(xs); color = PALETTE.ink, linewidth = 1.8)
lines!(ax, [x0, x0], [0, f(x0)]; color = PALETTE.grey, linestyle = :dash)
hlines!(ax, [u]; color = PALETTE.grey, linewidth = 0.8)
lines!(ax, xs, [f(x) > u ? u : NaN for x in xs]; color = PALETTE.blue, linewidth = 6,
       label = L"\text{slice } S_u = \{x : f(x) > u\}")
for (k, (l, r)) in enumerate(steps)
    yy = -0.035 - 0.022 * (k - 1)
    lines!(ax, [l, r], [yy, yy]; color = PALETTE.gold, linewidth = 1.6,
           label = k == 1 ? L"\text{stepping out, } w = 1" : nothing)
    scatter!(ax, [l, r], [yy, yy]; marker = :vline, markersize = 10, color = PALETTE.gold)
end
scatter!(ax, rej, fill(u, length(rej)); marker = :xcross, markersize = 14, color = PALETTE.red,
         label = "rejected, interval shrinks")
scatter!(ax, [y], [u]; markersize = 11, color = PALETTE.green, label = L"\text{accepted } x'")
scatter!(ax, [x0], [u]; markersize = 7, color = PALETTE.ink)
text!(ax, x0 + 0.08, 0.005; text = L"x")
lines!(ax, [-1.1, x0], [0.3, u]; color = PALETTE.grey, linewidth = 0.8)
text!(ax, -1.1, 0.3; text = L"u \sim \mathrm{U}(0, f(x))", align = (:left, :bottom))
xlims!(ax, -5, 5); ylims!(ax, -0.035 - 0.022 * length(steps), 0.5)
axislegend(ax; position = :rt)
save_fig(fig, @__DIR__, "one-update")

# ---- Figure 2: output for two interval widths
rng = MersenneTwister(42)
T = 20_000
fig = Figure(size = (780, 420))
for (i, (w, col)) in enumerate([(0.3, PALETTE.red), (4.0, PALETTE.blue)])
    x = run(-2.5, w, T, rng)
    ax = Axis(fig[i, 1]; ylabel = L"x_t", titlesize = 13,
              title = @sprintf("w = %s: fraction of draws with x > 0 is %.2f (target 0.45)", w, mean(x .> 0)),
              xlabel = i == 2 ? L"\text{iteration } t" : "")
    lines!(ax, 1:T, x; color = col, linewidth = 0.4)
    xlims!(ax, 0, T); ylims!(ax, -5, 5)
    i == 1 && hidexdecorations!(ax; grid = false)
    axh = Axis(fig[i, 2]; xlabel = i == 2 ? L"x" : "", titlesize = 13,
               title = i == 1 ? "histogram and target" : "")
    hist!(axh, x; bins = range(-5, 5; length = 61), normalization = :pdf, color = (col, 0.45))
    lines!(axh, xs, f.(xs); color = PALETTE.ink, linewidth = 1.5)
    xlims!(axh, -5, 5); ylims!(axh, 0, 0.62)
    i == 1 && hidexdecorations!(axh; grid = false)
end
colsize!(fig.layout, 1, Relative(0.63))
save_fig(fig, @__DIR__, "interval-width")
