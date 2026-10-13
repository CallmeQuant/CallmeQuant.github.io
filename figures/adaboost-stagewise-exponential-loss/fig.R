# AdaBoost with decision stumps on a disc-shaped class: decision region, error curves
# with the exponential-loss bound, and normalized margin distributions.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(dplyr) })

set.seed(2718)
R2 <- 0.55; N_TRAIN <- 200; N_TEST <- 5000; M <- 400; SNAPSHOTS <- c(5, 50, 400)

sample_xy <- function(n) {
  x <- matrix(runif(2 * n, -1, 1), ncol = 2)
  list(x = x, y = ifelse(rowSums(x^2) < R2, 1, -1))
}

fit_stump <- function(x, y, w) {
  best <- list(j = 1, t = 0, s = 1, err = Inf)
  for (j in 1:2) {
    o <- order(x[, j]); xs <- x[o, j]; ys <- y[o]; ws <- w[o]
    left_pos <- c(0, cumsum(ws * (ys == 1)))
    right_neg <- rev(c(0, cumsum(rev(ws * (ys == -1)))))
    err <- left_pos + right_neg
    n <- length(xs)
    cuts <- c(xs[1] - 1e-9, (xs[-n] + xs[-1]) / 2, xs[n] + 1e-9)
    for (s in c(1, -1)) {
      e <- if (s == 1) err else 1 - err
      k <- which.min(e)
      if (e[k] < best$err) best <- list(j = j, t = cuts[k], s = s, err = e[k])
    }
  }
  best
}

stump_predict <- function(x, st) st$s * ifelse(x[, st$j] > st$t, 1, -1)

adaboost <- function(x, y, rounds) {
  w <- rep(1 / length(y), length(y)); stumps <- vector("list", rounds)
  alphas <- Z <- numeric(rounds)
  for (m in seq_len(rounds)) {
    st <- fit_stump(x, y, w)
    a <- 0.5 * log((1 - st$err) / st$err)
    w <- w * exp(-a * y * stump_predict(x, st))
    Z[m] <- sum(w); w <- w / sum(w)
    stumps[[m]] <- st; alphas[m] <- a
  }
  list(stumps = stumps, alphas = alphas, Z = Z)
}

score <- function(x, fit, upto = length(fit$alphas)) {   # rows = observations, cols = rounds
  H <- sapply(fit$stumps[seq_len(upto)], function(st) stump_predict(x, st))
  t(apply(sweep(H, 2, fit$alphas[seq_len(upto)], `*`), 1, cumsum))
}

tr <- sample_xy(N_TRAIN); te <- sample_xy(N_TEST)
fit <- adaboost(tr$x, tr$y, M)
Ftr <- score(tr$x, fit); Fte <- score(te$x, fit)
curves <- bind_rows(
  data.frame(m = 1:M, value = cumprod(fit$Z), what = "bound ~ Z[1] * Z[2] * cdots * Z[M]"),
  data.frame(m = 1:M, value = colMeans(sign(Ftr) != tr$y), what = "'training error'"),
  data.frame(m = 1:M, value = colMeans(sign(Fte) != te$y), what = "'test error'"))
curves$what <- factor(curves$what, levels = unique(curves$what))

g <- seq(-1, 1, length.out = 201)
grid <- as.matrix(expand.grid(x1 = g, x2 = g))
Fg <- score(grid, fit, SNAPSHOTS[2])[, SNAPSHOTS[2]]
gd <- data.frame(x1 = grid[, 1], x2 = grid[, 2], F = Fg, cls = factor(sign(Fg)))
pts <- data.frame(x1 = tr$x[, 1], x2 = tr$x[, 2], y = factor(tr$y))
circ <- data.frame(t = seq(0, 2 * pi, length.out = 400)) |>
  mutate(x1 = sqrt(R2) * cos(t), x2 = sqrt(R2) * sin(t))

p1 <- ggplot(gd, aes(x1, x2)) +
  geom_raster(aes(fill = cls), show.legend = FALSE) +
  scale_fill_manual(values = c("-1" = "#f3e3dd", "1" = "#dfe5f2")) +
  geom_contour(aes(z = F), breaks = 0, colour = PALETTE[["ink"]], linewidth = 0.4) +
  geom_path(data = circ, linetype = "dashed", colour = PALETTE[["grey"]]) +
  geom_point(data = pts, aes(shape = y, colour = y), size = 1.3, show.legend = FALSE) +
  scale_shape_manual(values = c("-1" = 4, "1" = 16)) +
  scale_colour_manual(values = c("-1" = PALETTE[["red"]], "1" = PALETTE[["blue"]])) +
  coord_equal(expand = FALSE) +
  labs(x = expression(x[1]), y = expression(x[2]),
       title = bquote(sign ~ F[M] ~ "after" ~ M == .(SNAPSHOTS[2]) ~ "stumps")) +
  theme_blog() + theme(panel.grid.major = element_blank())

p2 <- ggplot(curves, aes(m, value, colour = what)) + geom_line(linewidth = 0.7) +
  scale_x_log10() + scale_y_continuous(limits = c(0, 1.02)) +
  scale_colour_manual(values = unname(PALETTE[c("gold", "blue", "red")]),
                      labels = scales::label_parse()) +
  labs(x = "number of rounds M", y = NULL, title = "error rates and the exponential-loss bound") +
  theme_blog() + theme(legend.position = c(1, 1), legend.justification = c(1, 1))
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.35)), here, "rounds", width = 7.6, height = 3.6)

marg <- bind_rows(lapply(SNAPSHOTS, function(Mm) {
  data.frame(margin = tr$y * Ftr[, Mm] / sum(fit$alphas[1:Mm]),
             lab = sprintf("M = %d (train err %.3f)", Mm, mean(sign(Ftr[, Mm]) != tr$y)))
}))
marg$lab <- factor(marg$lab, levels = unique(marg$lab))
p3 <- ggplot(marg, aes(margin, colour = lab)) +
  stat_ecdf(geom = "step", pad = TRUE, linewidth = 0.7) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "green", "violet")])) +
  coord_cartesian(xlim = c(-0.3, 0.6)) +
  labs(x = expression("normalized margin " * y[i] * f[M](x[i])), y = "fraction of training points") +
  theme_blog() + theme(legend.position = c(1, 0.04), legend.justification = c(1, 0))
save_fig(p3, here, "margins", width = 6.0, height = 3.3)
