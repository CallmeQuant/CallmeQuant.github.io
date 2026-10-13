# Shared plotting style for the figures on this site (Julia / CairoMakie).
#
# Usage inside figures/<slug>/fig.jl:
#   include(joinpath(@__DIR__, "..", "_style", "style.jl"))
#   set_blog_theme!()
#   fig = Figure(size = (640, 360))
#   ax = Axis(fig[1, 1]; xlabel = L"\theta", ylabel = "density")
#   ...
#   save_fig(fig, @__DIR__, "posterior")     # -> figures/<slug>/output/posterior.png
#
# Packages: CairoMakie (plus whatever a script lists in its header).

using CairoMakie

const PALETTE = (
    ink = colorant"#1a2233", blue = colorant"#2b4c9b", red = colorant"#b5482f",
    green = colorant"#2e7d5b", gold = colorant"#c2902a", violet = colorant"#6a4c93",
    grey = colorant"#8a93a4", light = colorant"#dde2ea",
)
const CYCLE = [PALETTE.blue, PALETTE.red, PALETTE.green, PALETTE.gold, PALETTE.violet, PALETTE.grey]

function blog_theme()
    base = Theme(
        fontsize = 15,
        textcolor = PALETTE.ink,
        palette = (color = CYCLE,),
        backgroundcolor = :white,
        Axis = (
            backgroundcolor = :white,
            topspinevisible = false, rightspinevisible = false,
            leftspinecolor = PALETTE.ink, bottomspinecolor = PALETTE.ink,
            spinewidth = 0.9,
            xgridcolor = PALETTE.light, ygridcolor = PALETTE.light,
            xgridwidth = 0.8, ygridwidth = 0.8,
            xtickcolor = PALETTE.ink, ytickcolor = PALETTE.ink,
            titlefont = :regular, titlealign = :left, titlesize = 15,
        ),
        Legend = (framevisible = false, labelsize = 13),
        Lines = (linewidth = 2.2,),
    )
    return merge(base, theme_latexfonts())
end

set_blog_theme!() = set_theme!(blog_theme())

function save_fig(fig, here::AbstractString, name::AbstractString; px_per_unit = 2)
    out = joinpath(here, "output")
    mkpath(out)
    path = joinpath(out, name * ".png")
    save(path, fig; px_per_unit = px_per_unit)
    return path
end
