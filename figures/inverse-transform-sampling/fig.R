# Generalized inverse of a distribution function with a flat stretch and a jump
# (note: inverse transform sampling). Mixture: 0.35 Unif(0,1) + 0.25 delta_{1.5} + 0.40 Unif(2,3).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

Fcdf <- function(x) ifelse(x < 0, 0, ifelse(x < 1, 0.35 * x, ifelse(x < 1.5, 0.35,
              ifelse(x < 2, 0.60, ifelse(x < 3, 0.60 + 0.40 * (x - 2), 1)))))
Finv <- function(u) ifelse(u <= 0.35, u / 0.35, ifelse(u <= 0.60, 1.5, 2 + (u - 0.60) / 0.40))

blue <- PALETTE[["blue"]]; red <- PALETTE[["red"]]
segs <- data.frame(x0 = c(-0.4, 0, 1, 1.5, 2, 3), x1 = c(0, 1, 1.5, 2, 3, 3.4),
                   y0 = c(0, 0, 0.35, 0.6, 0.6, 1), y1 = c(0, 0.35, 0.35, 0.6, 1, 1))
us <- c(0.2, 0.5, 0.8)
arr <- data.frame(u = us, x = Finv(us))
pa <- ggplot() +
  geom_segment(data = arr, aes(x = -0.4, xend = x, y = u, yend = u), colour = red, linetype = "dashed", linewidth = 0.4) +
  geom_segment(data = arr, aes(x = x, xend = x, y = u, yend = 0), colour = red, linewidth = 0.4,
               arrow = arrow(length = unit(0.12, "cm"))) +
  geom_segment(data = segs, aes(x = x0, xend = x1, y = y0, yend = y1), colour = blue, linewidth = 0.8) +
  geom_point(aes(x = 1.5, y = 0.35), shape = 21, fill = "white", colour = blue, size = 1.8) +
  geom_point(aes(x = 1.5, y = 0.6), colour = blue, size = 1.8) +
  scale_y_continuous(breaks = c(0, 0.2, 0.35, 0.5, 0.6, 0.8, 1), labels = c("0", "0.2", "0.35", "0.5", "0.6", "0.8", "1")) +
  coord_cartesian(xlim = c(-0.4, 3.4), ylim = c(0, 1.05), expand = FALSE) +
  labs(x = "x", y = NULL, title = "(a) distribution function F") + theme_blog()

qs <- rbind(data.frame(u = seq(0, 0.35, length.out = 50), piece = 1),
            data.frame(u = seq(0.35 + 1e-9, 0.6, length.out = 50), piece = 2),
            data.frame(u = seq(0.6 + 1e-9, 1, length.out = 50), piece = 3))
qs$q <- Finv(qs$u)
pb <- ggplot(qs, aes(u, q, group = piece)) +
  geom_line(colour = blue, linewidth = 0.8) +
  geom_point(data = data.frame(u = c(0.35, 0.6), q = c(1, 1.5)), aes(u, q), inherit.aes = FALSE, colour = blue, size = 1.8) +
  geom_point(data = data.frame(u = c(0.35, 0.6), q = c(1.5, 2)), aes(u, q), inherit.aes = FALSE,
             shape = 21, fill = "white", colour = blue, size = 1.8) +
  scale_x_continuous(breaks = c(0, 0.35, 0.6, 1), labels = c("0", "0.35", "0.6", "1")) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-0.1, 3.2)) +
  labs(x = "u", y = NULL, title = expression("(b) quantile function" ~ F^{-1})) + theme_blog()

set.seed(2023)
X <- sort(Finv(runif(400)))
xx <- seq(-0.4, 3.4, length.out = 1000)
ecd <- data.frame(x = c(-0.4, X, 3.4), y = c(0, seq_along(X) / length(X), 1))
pc <- ggplot() +
  geom_line(data = data.frame(x = xx, y = Fcdf(xx), what = "F"), aes(x, y, colour = what), linewidth = 2, alpha = 0.6) +
  geom_step(data = transform(ecd, what = "empirical, n = 400"), aes(x, y, colour = what), linewidth = 0.4) +
  scale_colour_manual(values = c("F" = PALETTE[["grey"]], "empirical, n = 400" = red), breaks = c("F", "empirical, n = 400")) +
  coord_cartesian(xlim = c(-0.4, 3.4), ylim = c(0, 1.05)) +
  labs(x = "x", y = NULL, title = expression("(c) draws" ~ F^{-1}*(U[i]))) + theme_blog() +
  theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

save_fig(pa + pb + pc, here, "generalized-inverse", width = 8.6, height = 3.1)
