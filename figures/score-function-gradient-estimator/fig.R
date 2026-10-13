# Variance of score-function and pathwise gradient estimators for d/dtheta E[X^2], X ~ N(theta, sigma^2).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

sigma <- 1
set.seed(12)
estimators <- function(theta, z) {
  x <- theta + sigma * z; score <- z / sigma
  list(sf = x^2 * score, mean = (x^2 - (theta^2 + sigma^2)) * score,
       opt = (x^2 - (theta^2 + 3 * sigma^2)) * score, path = 2 * x)
}
exact <- list(sf = function(t) t^4 / sigma^2 + 14 * t^2 + 15 * sigma^2,
              mean = function(t) 8 * t^2 + 10 * sigma^2,
              opt = function(t) 8 * t^2 + 6 * sigma^2,
              path = function(t) 4 * sigma^2 + 0 * t)
keys <- c("sf", "mean", "opt", "path")
names_k <- c(sf = "'score function, no baseline'", mean = "'baseline' ~ b == E * f(X)",
             opt = "'optimal baseline' ~ b^'*'", path = "'pathwise (reparameterization)'")
cols <- c(sf = PALETTE[["red"]], mean = PALETTE[["gold"]], opt = PALETTE[["green"]], path = PALETTE[["blue"]])

t <- seq(0, 4, length.out = 200); te <- seq(0, 4, by = 0.5); z <- rnorm(2e5)
lin <- do.call(rbind, lapply(keys, function(k) data.frame(t = t, v = exact[[k]](t), k = k)))
pts <- do.call(rbind, lapply(keys, function(k) data.frame(t = te, v = sapply(te, function(th) var(estimators(th, z)[[k]])), k = k)))
lin$k <- factor(lin$k, keys); pts$k <- factor(pts$k, keys)
pa <- ggplot(lin, aes(t, v, colour = k)) + geom_line(linewidth = 0.7) + geom_point(data = pts, size = 1.4) +
  scale_y_log10(limits = c(2, 2e4)) +
  scale_colour_manual(values = cols, labels = scales::label_parse()(names_k[keys])) +
  labs(x = expression(theta), y = "variance of one-sample estimator",
       title = expression(f(x) == x^2 * "," ~ X %~% N(theta, 1))) +
  theme_blog() + theme(legend.position = c(0.02, 0.99), legend.justification = c(0, 1), legend.text = element_text(size = 8))

theta <- 2; n <- 5000; z <- rnorm(n); est <- estimators(theta, z); m <- seq_len(n)
run <- do.call(rbind, lapply(c("sf", "opt", "path"), function(k) {
  half <- 2 * sqrt(exact[[k]](theta) / m)
  data.frame(m = m, avg = cumsum(est[[k]]) / m, lo = 2 * theta - half, hi = 2 * theta + half, k = k)
}))
run$k <- factor(run$k, keys)
pb <- ggplot(run, aes(m, avg, colour = k, fill = k)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.12, colour = NA) + geom_line(linewidth = 0.4) +
  geom_hline(yintercept = 2 * theta, linetype = 2, linewidth = 0.3) +
  scale_x_log10() + coord_cartesian(xlim = c(5, n), ylim = c(-4, 12)) +
  scale_colour_manual(values = cols, guide = "none") + scale_fill_manual(values = cols, guide = "none") +
  labs(x = "number of samples n", y = "running average of estimates",
       title = expression("Estimating" ~ nabla * L(2) == 4 * "," ~ "bands" ~ "" %+-% 2 ~ sd / sqrt(n))) +
  theme_blog()
save_fig(pa + pb + plot_layout(widths = c(1.1, 1)), here, "gradient-variance", width = 7.8, height = 3.5)
