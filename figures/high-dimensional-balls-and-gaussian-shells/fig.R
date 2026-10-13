# Geometry of high-dimensional balls and Gaussians.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1961)
d <- 1:50
logV <- d / 2 * log(pi) - lgamma(d / 2 + 1)
dv <- rbind(data.frame(d = d, v = exp(logV), k = "V_d (unit ball)"),
            data.frame(d = d, v = exp(logV - d * log(2)), k = "V_d / 2^d (ball / cube)"))
dv$k <- factor(dv$k, levels = unique(dv$k))
p1 <- ggplot(dv, aes(d, v, colour = k)) + geom_line(linewidth = 0.4) + geom_point(size = 0.9) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-28, 0, 4)) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  labs(x = "dimension d", y = NULL, title = "Volume of the unit ball") +
  theme_blog() + theme(legend.position = c(0, 0), legend.justification = c(0, 0))

eps <- seq(0, 0.5, length.out = 300); ds <- c(2, 10, 100, 1000)
dsh <- do.call(rbind, lapply(ds, function(dd) data.frame(e = eps, f = 1 - (1 - eps)^dd, k = paste0("d = ", dd))))
dsh$k <- factor(dsh$k, levels = paste0("d = ", ds))
p2 <- ggplot(dsh, aes(e, f, colour = k)) + geom_line(linewidth = 0.8) + scale_colour_manual(values = CYCLE[1:4]) +
  labs(x = "relative shell thickness", y = NULL, title = "Volume fraction with |x| >= 1 - eps") +
  theme_blog() + theme(legend.position = c(1, 0), legend.justification = c(1, 0))

s <- seq(0.001, 2.2, length.out = 800); dr <- c(1, 3, 10, 100, 1000)
dchi <- function(r, k) exp((k - 1) * log(r) - r^2 / 2 - (k / 2 - 1) * log(2) - lgamma(k / 2))
drad <- do.call(rbind, lapply(dr, function(dd) data.frame(s = s, f = dchi(s * sqrt(dd), dd) * sqrt(dd), k = paste0("d = ", dd))))
drad$k <- factor(drad$k, levels = paste0("d = ", dr))
p3 <- ggplot(drad, aes(s, f, colour = k)) + geom_line(linewidth = 0.8) + scale_colour_manual(values = CYCLE[1:5]) +
  coord_cartesian(ylim = c(0, 19.5)) +
  labs(x = "|Z| / sqrt(d)", y = NULL, title = "Radius of Z ~ N(0, I_d), rescaled") +
  theme_blog() + theme(legend.position = c(1, 1), legend.justification = c(1, 1))
save_fig(p1 + p2 + p3, here, "volumes-and-shells", width = 10.2, height = 3.3)

dang <- do.call(rbind, lapply(c(3, 30, 300), function(dd) {
  Z1 <- matrix(rnorm(20000 * dd), ncol = dd); Z2 <- matrix(rnorm(20000 * dd), ncol = dd)
  cs <- rowSums(Z1 * Z2) / sqrt(rowSums(Z1^2) * rowSums(Z2^2))
  data.frame(a = acos(cs) * 180 / pi, k = paste0("d = ", dd))
}))
dang$k <- factor(dang$k, levels = paste0("d = ", c(3, 30, 300)))
q1 <- ggplot(dang, aes(a, after_stat(density), fill = k)) +
  geom_histogram(breaks = seq(0, 180, 2), position = "identity", alpha = 0.5) +
  geom_vline(xintercept = 90, linetype = "dotted", colour = PALETTE[["ink"]]) +
  scale_fill_manual(values = CYCLE[1:3]) + scale_x_continuous(breaks = seq(0, 180, 45)) +
  labs(x = "angle between two Gaussian vectors (degrees)", y = "density (per degree)",
       title = "Random directions are nearly orthogonal") +
  theme_blog() + theme(legend.position = c(0, 1), legend.justification = c(0, 1))

dims <- unique(round(10^seq(0, 4, length.out = 25)))
dd_df <- do.call(rbind, lapply(dims, function(dd) {
  v <- replicate(10, { D <- dist(matrix(rnorm(100 * dd), ncol = dd)); (max(D) - min(D)) / min(D) })
  data.frame(d = dd, med = median(v), lo = min(v), hi = max(v))
}))
q2 <- ggplot(dd_df, aes(d)) + geom_ribbon(aes(ymin = lo, ymax = hi), fill = PALETTE[["light"]]) +
  geom_line(aes(y = med), colour = PALETTE[["blue"]]) + geom_point(aes(y = med), colour = PALETTE[["blue"]], size = 1) +
  geom_line(aes(y = 8 / sqrt(d)), colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.3) +
  scale_x_log10() + scale_y_log10() +
  labs(x = "dimension d", y = "(max - min) / min distance", title = "Pairwise distances of 100 Gaussian points") +
  theme_blog()
save_fig(q1 + q2, here, "angles-and-distances", width = 9.2, height = 3.3)
