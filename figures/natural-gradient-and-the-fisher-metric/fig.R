# Fisher geometry of N(mu, sigma^2): KL balls, and gradient versus natural gradient descent.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

kl <- function(m0, s0, m, s) log(s / s0) + (s0^2 + (m0 - m)^2) / (2 * s^2) - 0.5
xbar <- 1; s2 <- 1
loss <- function(m, s) log(s) + (s2 + (xbar - m)^2) / (2 * s^2)
grad <- function(th) c(-(xbar - th[1]) / th[2]^2, 1 / th[2] - (s2 + (xbar - th[1])^2) / th[2]^3)
start <- c(-2, 0.5); eta <- 0.05; K <- 100
run <- function(natural) {
  th <- start; P <- matrix(th, 1)
  for (k in 1:K) {
    d <- grad(th); if (natural) d <- c(th[2]^2, th[2]^2 / 2) * d
    th <- th - eta * d; P <- rbind(P, th)
  }
  P
}
Lstar <- loss(xbar, sqrt(s2))
P_gd <- run(FALSE); P_ng <- run(TRUE)
lab_gd <- sprintf("GD: gap %.1e", loss(P_gd[K + 1, 1], P_gd[K + 1, 2]) - Lstar)
lab_ng <- sprintf("natural GD: gap %.1e", loss(P_ng[K + 1, 1], P_ng[K + 1, 2]) - Lstar)

grid <- expand.grid(m = seq(-3, 3, length.out = 400), s = seq(0.02, 3.2, length.out = 400))
balls <- do.call(rbind, lapply(c(0.25, 0.5, 1, 2), function(s0) do.call(rbind, lapply(c(-2, 0, 2), function(m0)
  transform(grid, z = kl(m0, s0, m, s), id = paste(m0, s0))))))
centers <- expand.grid(m = c(-2, 0, 2), s = c(0.25, 0.5, 1, 2))
pa <- ggplot() +
  geom_contour(data = balls, aes(m, s, z = z, group = id), breaks = 0.05, colour = PALETTE[["blue"]], linewidth = 0.5) +
  geom_point(data = centers, aes(m, s), size = 0.6) +
  coord_equal(xlim = c(-3, 3), ylim = c(0, 3.2), expand = FALSE) +
  labs(x = expression(mu), y = expression(sigma),
       title = expression("(a) Balls" ~ {KL(p[theta[0]] * "||" * p[theta]) <= 0.05})) +
  theme_blog()

g2 <- expand.grid(m = seq(-2.6, 2, length.out = 300), s = seq(0.05, 4.8, length.out = 300))
g2$z <- loss(g2$m, g2$s) - Lstar
paths <- rbind(data.frame(m = P_gd[, 1], s = P_gd[, 2], w = lab_gd), data.frame(m = P_ng[, 1], s = P_ng[, 2], w = lab_ng))
paths$w <- factor(paths$w, levels = c(lab_gd, lab_ng))
pb <- ggplot() +
  geom_contour(data = g2, aes(m, s, z = z), breaks = exp(seq(log(0.01), log(8), length.out = 12)),
               colour = PALETTE[["grey"]], linewidth = 0.3) +
  geom_path(data = paths, aes(m, s, colour = w), linewidth = 0.55) +
  geom_point(data = paths, aes(m, s, colour = w), size = 0.7) +
  annotate("point", x = start[1], y = start[2], shape = 15, size = 2) +
  annotate("point", x = xbar, y = 1, shape = 8, size = 2.5) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) +
  coord_cartesian(xlim = c(-2.6, 2), ylim = c(0, 4.8), expand = FALSE) +
  labs(x = expression(mu), y = expression(sigma),
       title = expression("(b) Fitting" ~ N(mu, sigma^2) * ", step" ~ eta == 0.05 * ", 100 steps")) +
  theme_blog() + theme(legend.position = "bottom", legend.direction = "horizontal", legend.text = element_text(size = 7.5))
save_fig(pa + pb, here, "fisher-geometry", width = 7.8, height = 3.5)
