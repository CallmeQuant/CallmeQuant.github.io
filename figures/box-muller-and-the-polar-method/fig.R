# Box-Muller map and Marsaglia's polar method.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1958)
box_muller <- function(u1, u2) {
  r <- sqrt(-2 * log(u1)); cbind(r * cos(2 * pi * u2), r * sin(2 * pi * u2))
}

# ---------- Figure 1: the map acting on a grid ----------
lv1 <- c(0.02, 0.1, 0.3, 0.6, 0.9); lv2 <- (0:7) / 8
n_pts <- 1500
u1 <- 1 - runif(n_pts); u2 <- runif(n_pts)
z <- box_muller(u1, u2)
lab <- factor(paste0("u[1] == ", lv1), levels = paste0("u[1] == ", lv1))
tt <- seq(0, 2 * pi, length.out = 400)
circ <- do.call(rbind, lapply(seq_along(lv1), function(k) {
  r <- sqrt(-2 * log(lv1[k])); data.frame(x = r * cos(tt), y = r * sin(tt), lev = lab[k])
}))
rays <- data.frame(x = 0, y = 0, xend = 3.3 * cos(2 * pi * lv2), yend = 3.3 * sin(2 * pi * lv2))

p1 <- ggplot() +
  geom_point(aes(u1, u2), colour = PALETTE[["grey"]], alpha = 0.55, size = 0.35) +
  geom_hline(yintercept = lv2, colour = PALETTE[["ink"]], linewidth = 0.25, alpha = 0.6) +
  geom_vline(aes(xintercept = lv1, colour = lab), linewidth = 0.7, show.legend = FALSE) +
  scale_colour_manual(values = CYCLE[1:5]) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(x = expression(u[1]), y = expression(u[2]), title = expression("Uniform pairs " * (U[1] * "," ~ U[2]))) +
  theme_blog() + theme(panel.grid.major = element_blank())
p2 <- ggplot() +
  geom_point(aes(z[, 1], z[, 2]), colour = PALETTE[["grey"]], alpha = 0.55, size = 0.35) +
  geom_segment(data = rays, aes(x, y, xend = xend, yend = yend), colour = PALETTE[["ink"]], linewidth = 0.25, alpha = 0.6) +
  geom_path(data = circ, aes(x, y, colour = lev), linewidth = 0.7) +
  scale_colour_manual(values = CYCLE[1:5], labels = function(l) parse(text = l)) +
  coord_equal(xlim = c(-3.4, 3.4), ylim = c(-3.4, 3.4)) +
  labs(x = expression(z[1]), y = expression(z[2]), title = expression("Image " * (Z[1] * "," ~ Z[2]))) +
  theme_blog() + theme(panel.grid.major = element_blank(), legend.position = "right")
save_fig(p1 + p2, here, "box-muller-map", width = 7.8, height = 3.8)

# ---------- Figure 2: polar method ----------
m <- 1200
v <- matrix(runif(2 * m, -1, 1), ncol = 2); s <- rowSums(v^2)
acc <- s > 0 & s < 1
N <- 200000
w <- matrix(runif(2 * round(1.4 * N), -1, 1), ncol = 2); sw <- rowSums(w^2)
keep <- which(sw > 0 & sw < 1)[1:N]; w <- w[keep, ]; sw <- sw[keep]
zz <- w * sqrt(-2 * log(sw) / sw)

q1 <- ggplot(data.frame(v1 = v[, 1], v2 = v[, 2], a = ifelse(acc, "accepted", "rejected"))) +
  geom_point(aes(v1, v2, colour = a), size = 0.4, show.legend = FALSE) +
  annotate("path", x = cos(tt), y = sin(tt), colour = PALETTE[["ink"]], linewidth = 0.4) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  coord_equal(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
  labs(x = expression(v[1]), y = expression(v[2]),
       title = bquote("Accept rate" ~ .(sprintf("%.3f", mean(acc))) ~ (pi/4 %~~% 0.785))) +
  theme_blog() + theme(panel.grid.major = element_blank(), plot.title = element_text(size = 9))
q2 <- ggplot(data.frame(s = sw)) +
  geom_histogram(aes(s, after_stat(density)), breaks = seq(0, 1, length.out = 41),
                 fill = PALETTE[["grey"]], alpha = 0.45, colour = "white", linewidth = 0.2) +
  geom_hline(yintercept = 1, colour = PALETTE[["ink"]]) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.35)) +
  labs(x = expression(S == V[1]^2 + V[2]^2 ~ "(accepted)"), y = "density", title = expression(S %~% "Uniform(0,1)")) + theme_blog()
xx <- seq(-4, 4, length.out = 400)
q3 <- ggplot(data.frame(z = zz[, 1])) +
  geom_histogram(aes(z, after_stat(density)), breaks = seq(-4, 4, length.out = 61),
                 fill = PALETTE[["grey"]], alpha = 0.45, colour = "white", linewidth = 0.2) +
  annotate("line", x = xx, y = dnorm(xx), colour = PALETTE[["blue"]], linewidth = 0.8) +
  coord_cartesian(xlim = c(-4, 4), ylim = c(0, 0.47)) +
  labs(x = expression(Z[1]), y = "density", title = "Output vs. N(0,1) density") + theme_blog()
save_fig(q1 + q2 + q3 + plot_layout(widths = c(1, 1.15, 1.15)), here, "polar-method", width = 8.8, height = 3.1)
