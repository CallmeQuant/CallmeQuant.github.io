# Conjugate Bayesian inference for the normal model.
# (a) posterior of the mean with known variance; (b) Student-t marginal posterior under the
# normal-inverse-gamma prior versus a normal density with the same location and scale.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(2023)
mu_true <- 1; sigma <- 2
x <- rnorm(50, mu_true, sigma)

m <- 0; tau <- 1.5
grid <- seq(-3, 4, length.out = 600)
curves <- data.frame(mu = grid, dens = dnorm(grid, m, tau), lab = "prior")
for (n in c(2, 10, 50)) {
  prec <- 1 / tau^2 + n / sigma^2
  mn <- (m / tau^2 + sum(x[1:n]) / sigma^2) / prec
  curves <- rbind(curves, data.frame(mu = grid, dens = dnorm(grid, mn, prec^-0.5),
                                     lab = paste0("n = ", n)))
}
curves$lab <- factor(curves$lab, levels = c("prior", "n = 2", "n = 10", "n = 50"))
pa <- ggplot(curves, aes(mu, dens, colour = lab, linetype = lab)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = mu_true, linetype = "dotted", colour = PALETTE[["ink"]], linewidth = 0.4) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "gold", "red", "blue")])) +
  scale_linetype_manual(values = c("dashed", "solid", "solid", "solid")) +
  labs(x = expression(mu), y = "density", title = expression("(a) known " * sigma^2 * ": posterior of " * mu)) +
  theme_blog() + theme(legend.position = c(0.2, 0.72), legend.justification = c(0.5, 0.5))

m0 <- 0; k0 <- 1; a0 <- 1; b0 <- 1; n <- 5
xs <- x[1:n]; xb <- mean(xs)
kn <- k0 + n; mn <- (k0 * m0 + n * xb) / kn
an <- a0 + n / 2
bn <- b0 + 0.5 * sum((xs - xb)^2) + k0 * n * (xb - m0)^2 / (2 * kn)
sc <- sqrt(bn / (an * kn))
g2 <- seq(mn - 8 * sc, mn + 8 * sc, length.out = 600)
d2 <- rbind(
  data.frame(mu = g2, dens = dt((g2 - mn) / sc, 2 * an) / sc, lab = sprintf("Student t, %d d.f.", 2 * an)),
  data.frame(mu = g2, dens = dnorm(g2, mn, sc), lab = "normal, same scale"))
d2$lab <- factor(d2$lab, levels = unique(d2$lab))
pb <- ggplot(d2, aes(mu, dens, colour = lab, linetype = lab)) +
  geom_line(linewidth = 0.8) +
  scale_y_log10(limits = c(1e-5, 2), labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  labs(x = expression(mu), y = NULL, title = expression("(b) unknown " * sigma^2 * ", n = 5 (log scale)")) +
  theme_blog() + theme(legend.position = "bottom", legend.justification = "center")

save_fig(pa + pb, here, "posteriors", width = 7.4, height = 3.4)
