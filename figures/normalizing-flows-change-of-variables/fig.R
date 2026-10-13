# Normalizing flows: change of variables in one dimension (three examples) and two affine
# coupling layers applied to a standard Gaussian in the plane.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(dplyr) })

set.seed(1505)
n <- 100000
hist_layer <- function(x, lo, hi) {
  geom_histogram(data = data.frame(x = x[x >= lo & x <= hi]), aes(x, after_stat(count) / (n * (hi - lo) / 80)),
                 binwidth = (hi - lo) / 80, boundary = lo, fill = PALETTE[["light"]], colour = "white", linewidth = 0.1,
                 inherit.aes = FALSE)
}
leg <- theme(legend.position = c(1, 1), legend.justification = c(1, 1),
             plot.margin = margin(5, 14, 5, 5))

# (a) X = U^2, U ~ Uniform(0, 1)
xs <- seq(1e-3, 1, length.out = 400)
pa <- ggplot() + hist_layer(runif(n)^2, 0, 1) +
  geom_line(data = data.frame(x = xs, y = 1 / (2 * sqrt(xs)), lab = "1/(2*sqrt(x))"),
            aes(x, y, colour = lab), linewidth = 0.7) +
  scale_colour_manual(values = PALETTE[["blue"]], labels = scales::label_parse()) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 5), expand = FALSE) +
  labs(x = "x", y = "density", title = "(a) U ~ Unif(0,1),  X = U^2") + theme_blog() + leg

# (b) X = U + 2 tanh(2U), U ~ N(0, 1)
Tm <- function(u) u + 2 * tanh(2 * u)
dT <- function(u) 1 + 4 / cosh(2 * u)^2
us <- seq(-6, 6, length.out = 800)
cb <- bind_rows(data.frame(x = Tm(us), y = dnorm(us) / dT(us), lab = "phi(u) / T * minute(u)"),
                data.frame(x = us, y = dnorm(us), lab = "base ~ phi"))
cb$lab <- factor(cb$lab, levels = unique(cb$lab))
pb <- ggplot() + hist_layer(Tm(rnorm(n)), -6, 6) +
  geom_line(data = cb, aes(x, y, colour = lab, linetype = lab), linewidth = 0.7) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "grey")]), labels = scales::label_parse()) +
  scale_linetype_manual(values = c("solid", "dashed"), labels = scales::label_parse()) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(0, 0.55), expand = FALSE) +
  labs(x = "x", y = NULL, title = "(b) X = U + 2tanh(2U), U ~ N(0,1)") + theme_blog() + leg

# (c) X = U^2, U ~ N(0, 1)
xs <- seq(0.02, 5, length.out = 400)
cc <- bind_rows(data.frame(x = xs, y = dchisq(xs, 1), lab = "sum over both preimages"),
                data.frame(x = xs, y = dnorm(sqrt(xs)) / (2 * sqrt(xs)), lab = "branch u = +sqrt(x) only"))
cc$lab <- factor(cc$lab, levels = unique(cc$lab))
pc <- ggplot() + hist_layer(rnorm(n)^2, 0, 5) +
  geom_line(data = cc, aes(x, y, colour = lab, linetype = lab), linewidth = 0.7) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  coord_cartesian(xlim = c(0, 5), ylim = c(0, 1.5), expand = FALSE) +
  labs(x = "x", y = NULL, title = "(c) X = U^2,  U ~ N(0,1)") + theme_blog() + leg
save_fig(pa + pb + pc, here, "one-dim", width = 8.8, height = 3.0)

# ------------------------------------------------------------------- coupling layers
s1 <- function(a) 0.5 * tanh(a); t1 <- function(a) 0.6 * a^2 - 0.6
s2 <- function(b) -0.3 * tanh(b); t2 <- function(b) 0.9 * sin(1.2 * b)
layer1 <- function(z) list(z[[1]], z[[2]] * exp(s1(z[[1]])) + t1(z[[1]]))
layer2 <- function(z) list(z[[1]] * exp(s2(z[[2]])) + t2(z[[2]]), z[[2]])
apply_stage <- function(z, stage) { if (stage >= 1) z <- layer1(z); if (stage >= 2) z <- layer2(z); z }
logp_stage <- function(stage, y1, y2) {
  ld <- 0
  if (stage >= 2) { y1 <- (y1 - t2(y2)) * exp(-s2(y2)); ld <- ld - s2(y2) }
  if (stage >= 1) { ld <- ld - s1(y1); y2 <- (y2 - t1(y1)) * exp(-s1(y1)) }
  dnorm(y1, log = TRUE) + dnorm(y2, log = TRUE) + ld
}

u <- matrix(rnorm(4000), ncol = 2)
g <- seq(-4, 4, length.out = 301)
grid <- expand.grid(x1 = g, x2 = g)
tt <- seq(-2.6, 2.6, length.out = 200)
titles <- c("base N(0, I)", "after coupling layer 1", "after coupling layer 2")
panels <- lapply(0:2, function(stage) {
  gl <- bind_rows(lapply(seq(-2, 2, by = 1), function(cc) bind_rows(
    { z <- apply_stage(list(rep(cc, 200), tt), stage); data.frame(x1 = z[[1]], x2 = z[[2]], id = paste("v", cc)) },
    { z <- apply_stage(list(tt, rep(cc, 200)), stage); data.frame(x1 = z[[1]], x2 = z[[2]], id = paste("h", cc)) })))
  z <- apply_stage(list(u[, 1], u[, 2]), stage)
  dens <- grid |> mutate(p = exp(logp_stage(stage, x1, x2)))
  ggplot() +
    geom_path(data = gl, aes(x1, x2, group = id), colour = PALETTE[["grey"]], linewidth = 0.25, alpha = 0.8) +
    geom_point(data = data.frame(x1 = z[[1]], x2 = z[[2]]), aes(x1, x2), colour = PALETTE[["blue"]],
               size = 0.25, alpha = 0.35) +
    geom_contour(data = dens, aes(x1, x2, z = p), breaks = c(0.01, 0.04, 0.1), colour = PALETTE[["red"]],
                 linewidth = 0.4) +
    coord_equal(xlim = c(-3.6, 3.6), ylim = c(-3.6, 4.1)) +
    labs(x = expression(x[1]), y = if (stage == 0) expression(x[2]) else NULL, title = titles[stage + 1]) +
    theme_blog() + theme(panel.grid.major = element_blank())
})
save_fig(wrap_plots(panels, nrow = 1), here, "coupling", width = 8.8, height = 3.2)
