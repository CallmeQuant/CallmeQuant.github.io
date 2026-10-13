# Mean squared error of SS/d for d = n-1, n, n+1 and the MSE-optimal divisor, relative to d = n-1
# (note: Bessel's correction). Exact formulas.
# Packages: CairoMakie
include(joinpath(@__DIR__, "..", "_style", "style.jl"))

function mse(d, n, kappa)
    ess2 = (n - 1)^2 * (kappa / n - (n - 3) / (n * (n - 1)) + 1)
    return ess2 / d^2 - 2 * (n - 1) / d + 1
end

set_blog_theme!()
n = collect(2:30)
cases = [(1, L"\kappa = 1\ \text{(symmetric }\pm 1\text{)}"),
         (3, L"\kappa = 3\ \text{(normal)}"),
         (9, L"\kappa = 9\ \text{(exponential)}")]

fig = Figure(size = (820, 310))
axs = Axis[]
for (j, (kappa, title)) in enumerate(cases)
    ax = Axis(fig[1, j]; title = title, xlabel = L"\text{sample size } n",
              ylabel = j == 1 ? L"\mathrm{MSE}(d)\,/\,\mathrm{MSE}(n-1)" : "")
    base = mse.(n .- 1, n, kappa)
    dstar = n .+ 1 .+ (n .- 1) .* (kappa - 3) ./ n
    hlines!(ax, [1.0]; color = PALETTE.grey, linewidth = 1, linestyle = :dash)
    lines!(ax, n, mse.(n, n, kappa) ./ base; color = PALETTE.blue, label = L"d = n")
    lines!(ax, n, mse.(n .+ 1, n, kappa) ./ base; color = PALETTE.red, label = L"d = n+1")
    lines!(ax, n, mse.(dstar, n, kappa) ./ base; color = PALETTE.ink, linewidth = 1.4,
           linestyle = :dot, label = L"d = d^\ast")
    xlims!(ax, 2, 30); ylims!(ax, 0, 3.2)
    j > 1 && hideydecorations!(ax; grid = false)
    push!(axs, ax)
end
linkyaxes!(axs...)
axislegend(axs[3]; position = :rt)
save_fig(fig, @__DIR__, "mse-by-divisor")
