# Shared plotting style for the figures on this site (R / ggplot2).
#
# Usage inside figures/<slug>/fig.R (run from anywhere):
#   args <- commandArgs(trailingOnly = FALSE)
#   here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
#   source(file.path(here, "..", "_style", "style.R"))
#   p <- ggplot(...) + theme_blog()
#   save_fig(p, here, "posterior", width = 6.4, height = 3.6)
#
# Output goes to figures/<slug>/output/ (not tracked by git).

suppressPackageStartupMessages({
  library(ggplot2)
})

PALETTE <- c(
  ink = "#1a2233", blue = "#2b4c9b", red = "#b5482f", green = "#2e7d5b",
  gold = "#c2902a", violet = "#6a4c93", grey = "#8a93a4", light = "#dde2ea"
)
CYCLE <- unname(PALETTE[c("blue", "red", "green", "gold", "violet", "grey")])

theme_blog <- function(base_size = 11, base_family = "serif") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      text = element_text(colour = PALETTE[["ink"]]),
      axis.text = element_text(colour = PALETTE[["ink"]], size = rel(0.86)),
      axis.line = element_line(colour = PALETTE[["ink"]], linewidth = 0.35),
      axis.ticks = element_line(colour = PALETTE[["ink"]], linewidth = 0.3),
      panel.grid.major = element_line(colour = PALETTE[["light"]], linewidth = 0.3),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = rel(1.04), hjust = 0, margin = margin(b = 6)),
      strip.text = element_text(size = rel(0.95), margin = margin(b = 4)),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      legend.key.width = unit(1.6, "lines"),
      legend.text = element_text(hjust = 0, size = rel(0.86)),
      panel.spacing = unit(1.2, "lines"),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(6, 8, 6, 6)
    )
}

scale_colour_blog <- function(...) scale_colour_manual(values = CYCLE, ...)
scale_fill_blog <- function(...) scale_fill_manual(values = CYCLE, ...)

save_fig <- function(plot, here, name, width = 6.4, height = 3.6, dpi = 220) {
  out <- file.path(here, "output")
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(out, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = height, dpi = dpi, bg = "white")
  invisible(path)
}
