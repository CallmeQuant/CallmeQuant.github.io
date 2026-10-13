# Tweedie distributions with 1 < p < 2 as compound Poisson-gamma laws.
# Left: continuous part of the density (mu = 1, phi = 1). Right: sample variance against mean, log-log.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
set.seed(1984)

cpg <- function(mu, phi, p) list(lam = mu^(2 - p) / (phi * (2 - p)), alpha = (2 - p) / (p - 1), gam = phi * (p - 1) * mu^(p - 1))
dens <- function(y, mu, phi, p, nmax = 400) {
  q <- cpg(mu, phi, p); n <- 1:nmax
  sapply(y, function(v) sum(dpois(n, q$lam) * dgamma(v, shape = n * q$alpha, scale = q$gam)))
}
rtw <- function(size, mu, phi, p) {
  q <- cpg(mu, phi, p); N <- rpois(size, q$lam)
  ifelse(N > 0, rgamma(size, shape = pmax(N, 1) * q$alpha, scale = q$gam), 0)
}

y <- seq(0.005, 4, length.out = 800); ps <- c(1.1, 1.3, 1.5, 1.7, 1.9)
labs_p <- sapply(ps, function(p) { p0 <- exp(-cpg(1, 1, p)$lam)
  sprintf("p = %.1f:  P(Y = 0) = %s", p, if (p0 >= 0.01) sprintf("%.2f", p0) else sprintf("%.1e", p0)) })
d1 <- do.call(rbind, lapply(seq_along(ps), function(i) data.frame(y = y, d = dens(y, 1, 1, ps[i]), p = labs_p[i])))
d1$p <- factor(d1$p, levels = labs_p)
g1 <- ggplot(d1, aes(y, d, colour = p)) + geom_line(linewidth = 0.7) +
  scale_colour_manual(values = CYCLE[1:5]) +
  coord_cartesian(xlim = c(0, 4), ylim = c(0, 1)) +
  guides(colour = guide_legend(ncol = 1)) +
  labs(x = "y", y = "density of the continuous part", title = expression(mu == 1 * "," ~ phi == 1)) +
  theme_blog() + theme(legend.text = element_text(size = 7.5), legend.position = c(0.98, 0.98), legend.justification = c(1, 1))

mus <- 10^seq(-1, 1, length.out = 9); pv <- c(1.2, 1.5, 1.8)
d2 <- do.call(rbind, lapply(pv, function(p) data.frame(mu = mus, v = sapply(mus, function(m) var(rtw(4000, m, 1, p))),
                                                      p = sprintf("Var Y = mu^%.1f", p))))
l2 <- do.call(rbind, lapply(pv, function(p) data.frame(mu = mus, v = mus^p, p = sprintf("Var Y = mu^%.1f", p))))
g2 <- ggplot(d2, aes(mu, v, colour = p)) +
  geom_line(data = l2, linewidth = 0.6) + geom_point(shape = 21, fill = "white", size = 2, stroke = 0.9) +
  scale_x_log10() + scale_y_log10() +
  scale_colour_manual(values = CYCLE[c(1, 3, 5)]) +
  guides(colour = guide_legend(ncol = 1)) +
  labs(x = "mean mu (log scale)", y = "variance (log scale)", title = "Sample variances, n = 4000 per point") +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
save_fig(g1 + g2 + plot_layout(widths = c(1.1, 1)), here, "tweedie-cpg", width = 7.9, height = 4.2)
