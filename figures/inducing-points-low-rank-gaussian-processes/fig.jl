# Inducing-point approximations to Gaussian process regression: SoR, DTC and FITC predictions against
# the exact posterior, and their accuracy as the number of inducing points grows.
# Packages: CairoMakie, Random, LinearAlgebra, Statistics, Distributions
include(joinpath(@__DIR__, "..", "_style", "style.jl"))
using Random, LinearAlgebra, Statistics, Distributions

set_blog_theme!()
rng = MersenneTwister(2005)
const ELL, SF, SN = 0.8, 1.0, 0.3
kern(a, b) = SF^2 .* exp.(-0.5 .* (a .- b') .^ 2 ./ ELL^2)
f_true(x) = sin(x) + 0.5 * sin(3x)
n = 300
X = vcat(4 .* rand(rng, n ÷ 2), 6 .+ 4 .* rand(rng, n ÷ 2))
y = f_true.(X) .+ SN .* randn(rng, n)
xs = collect(range(-2.5, 12.5; length = 500))
Cy = cholesky(Symmetric(kern(X, X) + SN^2 * I))

function exact(xq)
    Ks = kern(xq, X)
    V = Cy.L \ Ks'
    return Ks * (Cy \ y), SF^2 .- vec(sum(V .^ 2; dims = 1))
end
function sparse(xq, Z, method)
    m = length(Z)
    Kuu = kern(Z, Z) + 1e-8I
    Kuf = kern(Z, X)
    Ksu = kern(xq, Z)
    Cu = cholesky(Symmetric(Kuu))
    lam = method == "FITC" ? SF^2 .- vec(sum(Kuf .* (Cu \ Kuf); dims = 1)) .+ SN^2 : fill(SN^2, n)
    Ca = cholesky(Symmetric(Kuu + (Kuf ./ lam') * Kuf'))
    mean_ = Ksu * (Ca \ (Kuf * (y ./ lam)))
    vw = vec(sum(Ksu .* (Ca \ Ksu')'; dims = 2))
    var_ = method == "SoR" ? vw : SF^2 .- vec(sum(Ksu .* (Cu \ Ksu')'; dims = 2)) .+ vw
    return mean_, var_
end

Z = collect(range(0.3, 9.7; length = 12))
fig = Figure(size = (780, 490))
panels = [("exact posterior", ""), ("SoR", "SoR"), ("DTC", "DTC"), ("FITC", "FITC")]
axs = Axis[]
for (j, (ttl, meth)) in enumerate(panels)
    r, c = fldmod1(j, 2)
    ax = Axis(fig[r, c]; title = ttl, xlabel = r == 2 ? L"x" : "", ylabel = c == 1 ? L"f(x)" : "")
    mu, v = meth == "" ? exact(xs) : sparse(xs, Z, meth)
    sd = sqrt.(max.(v, 0))
    scatter!(ax, X, y; color = (PALETTE.grey, 0.6), markersize = 3)
    band!(ax, xs, mu .- 1.96 .* sd, mu .+ 1.96 .* sd; color = (PALETTE.blue, 0.18))
    lines!(ax, xs, mu; color = PALETTE.blue, linewidth = 2)
    lines!(ax, xs, f_true.(xs); color = PALETTE.ink, linestyle = :dash, linewidth = 1)
    meth == "" || scatter!(ax, Z, fill(-2.55, length(Z)); marker = :utriangle, color = PALETTE.red, markersize = 9)
    xlims!(ax, -2.5, 12.5); ylims!(ax, -2.7, 2.7)
    push!(axs, ax)
end
linkaxes!(axs...)
hidexdecorations!(axs[1]; grid = false); hidexdecorations!(axs[2]; grid = false)
hideydecorations!(axs[2]; grid = false); hideydecorations!(axs[4]; grid = false)
save_fig(fig, @__DIR__, "sparse-predictions")

xt = collect(range(-2.5, 12.5; length = 400))
yt = f_true.(xt) .+ SN .* randn(rng, length(xt))
mex, vex = exact(xt)
lpd_ex = mean(logpdf.(Normal.(mex, sqrt.(vex .+ SN^2)), yt))
ms = collect(2:40)
cols = Dict("SoR" => PALETTE.red, "DTC" => PALETTE.blue, "FITC" => PALETTE.green)
fig = Figure(size = (780, 310))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = L"\text{number of inducing points } m", ylabel = "RMS difference of sd",
           title = L"\text{Posterior sd of } f \text{ vs. exact}")
ax2 = Axis(fig[1, 2]; xlabel = L"\text{number of inducing points } m", ylabel = "mean log predictive density",
           title = "Held-out predictive density")
for k in ("SoR", "DTC", "FITC")
    sde, lpd = Float64[], Float64[]
    for m in ms
        mu, v = sparse(xt, collect(range(0.3, 9.7; length = m)), k)
        push!(sde, sqrt(mean((sqrt.(max.(v, 0)) .- sqrt.(vex)) .^ 2)))
        push!(lpd, mean(logpdf.(Normal.(mu, sqrt.(max.(v, 0) .+ SN^2)), yt)))
    end
    ls = k == "DTC" ? :dash : :solid
    lines!(ax1, ms, sde; color = cols[k], linestyle = ls, label = k)
    lines!(ax2, ms, lpd; color = cols[k], linestyle = ls, label = k)
end
hlines!(ax2, [lpd_ex]; color = PALETTE.ink, linestyle = :dot, label = "exact GP")
xlims!(ax1, 2, 40); xlims!(ax2, 2, 40); ylims!(ax2, -4, 0)
axislegend(ax1; position = :rc); axislegend(ax2; position = :rb)
save_fig(fig, @__DIR__, "accuracy-vs-m")
