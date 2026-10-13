# Wiener process, Brownian motion with drift, an Ornstein-Uhlenbeck (Ito) process;
# variation of Brownian paths on finer grids.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1923)
T <- 4; N <- 2000; paths <- 5; dt <- T / N
t <- seq(0, T, length.out = N + 1)
dW <- matrix(rnorm(paths * N, sd = sqrt(dt)), nrow = paths)
W <- cbind(0, t(apply(dW, 1, cumsum)))
a <- 0.75; b <- 0.6; kappa <- 2; sigma <- 0.8; x0 <- 1.5
X <- sweep(b * W, 2, a * t, "+")
Y <- matrix(0, paths, N + 1); Y[, 1] <- x0
for (k in 1:N) Y[, k + 1] <- Y[, k] - kappa * Y[, k] * dt + sigma * dW[, k]

long <- function(M) data.frame(t = rep(t, each = paths), y = as.vector(M), path = factor(rep(1:paths, N + 1)))
panel <- function(M, lo, mid, hi, title) {
  ggplot() +
    geom_ribbon(aes(x = t, ymin = lo, ymax = hi), fill = PALETTE[["light"]]) +
    geom_line(aes(t, mid), colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.4) +
    geom_line(data = long(M), aes(t, y, colour = path), linewidth = 0.3, show.legend = FALSE) +
    scale_colour_manual(values = CYCLE[1:5]) +
    scale_x_continuous(limits = c(0, T), expand = c(0, 0)) +
    labs(x = "t", y = NULL, title = title) + theme_blog()
}
m <- x0 * exp(-kappa * t); s <- sigma * sqrt((1 - exp(-2 * kappa * t)) / (2 * kappa))
p1 <- panel(W, -2 * sqrt(t), 0 * t, 2 * sqrt(t), expression(W[t] * ": Wiener process"))
p2 <- panel(X, a * t - 2 * b * sqrt(t), a * t, a * t + 2 * b * sqrt(t), expression(X[t] == 0.75 * t + 0.6 * W[t]))
p3 <- panel(Y, m - 2 * s, m, m + 2 * s, expression(dY[t] == -2 * Y[t] * dt + 0.8 * dW[t]))
save_fig(p1 + p2 + p3, here, "three-processes", width = 9.8, height = 3.3)

K <- 2^20; levels <- 2:20
inc <- matrix(rnorm(3 * K, sd = sqrt(1 / K)), nrow = 3)
d <- do.call(rbind, lapply(1:3, function(i) {
  path <- c(0, cumsum(inc[i, ]))
  do.call(rbind, lapply(levels, function(l) {
    dd <- diff(path[seq(1, K + 1, by = 2^(20 - l))])
    data.frame(n = 2^l, qv = sum(dd^2), tv = sum(abs(dd)), path = factor(i))
  }))
}))
q1 <- ggplot(d, aes(n, qv, colour = path)) + geom_line(linewidth = 0.4) + geom_point(size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = PALETTE[["ink"]], linewidth = 0.3) +
  scale_x_continuous(trans = "log2", breaks = 2^seq(3, 19, 4), labels = scales::label_math(2^.x, format = log2)) +
  scale_colour_manual(values = CYCLE[1:3], guide = "none") + coord_cartesian(ylim = c(0, 2.6)) +
  labs(x = "number of intervals n", y = NULL, title = "Sum of squared increments on [0, 1]") + theme_blog()
q2 <- ggplot(d, aes(n, tv, colour = path)) + geom_line(linewidth = 0.4) + geom_point(size = 1) +
  geom_line(data = data.frame(n = 2^levels), aes(n, sqrt(2 * n / pi)), inherit.aes = FALSE,
            colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.3) +
  annotate("text", x = 2^3, y = 500, label = "sqrt(2*n/pi)", parse = TRUE, hjust = 0, size = 3.4) +
  scale_x_continuous(trans = "log2", breaks = 2^seq(3, 19, 4), labels = scales::label_math(2^.x, format = log2)) +
  scale_y_log10() + scale_colour_manual(values = CYCLE[1:3], guide = "none") +
  labs(x = "number of intervals n", y = NULL, title = "Sum of absolute increments on [0, 1]") + theme_blog()
save_fig(q1 + q2, here, "variation", width = 8.2, height = 3.1)
