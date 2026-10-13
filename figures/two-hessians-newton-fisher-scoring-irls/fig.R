# Newton's method versus Fisher scoring for a gamma GLM with identity link.
# Left: log-likelihood contours, region where the observed information is not positive definite,
# iterates from two starting points. Right: error against iteration near the MLE.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
set.seed(1972)
n <- 60; NU <- 2
x <- runif(n); X <- cbind(1, x); y <- rgamma(n, shape = NU, scale = (1 + 2 * x) / NU)

loglik <- function(b) { m <- as.vector(X %*% b); if (any(m <= 0)) -Inf else NU * sum(-y / m - log(m)) }
score <- function(b) { m <- as.vector(X %*% b); NU * as.vector(t(X) %*% ((y - m) / m^2)) }
obs_info <- function(b) { m <- as.vector(X %*% b); NU * t(X * ((2 * y - m) / m^3)) %*% X }
exp_info <- function(b) { m <- as.vector(X %*% b); NU * t(X / m^2) %*% X }
iterate <- function(b, method, iters) {
  path <- matrix(b, nrow = 1)
  for (i in seq_len(iters)) {
    b <- path[nrow(path), ]
    M <- if (method == "newton") obs_info(b) else exp_info(b)
    step <- solve(M, score(b)); s <- 1
    while (!is.finite(loglik(b + s * step))) s <- s / 2
    path <- rbind(path, b + s * step)
  }
  path
}
bhat <- iterate(c(1, 2), "fisher", 60)[61, ]

g <- expand.grid(b0 = seq(0.02, 8, length.out = 200), b1 = seq(-8, 8, length.out = 200))
g$L <- apply(g, 1, function(r) loglik(r[1:2]))
g$dev <- loglik(bhat) - g$L
g$notnd <- apply(g, 1, function(r) is.finite(loglik(r[1:2])) && min(eigen(obs_info(r[1:2]), symmetric = TRUE)$values) <= 0)
paths <- do.call(rbind, lapply(list(c(6, 6), c(4, 0.5)), function(st) do.call(rbind, lapply(c("newton", "fisher"), function(m) {
  P <- iterate(st, m, 12)
  data.frame(b0 = P[, 1], b1 = P[, 2], method = ifelse(m == "newton", "Newton", "Fisher scoring"),
             start = paste(st, collapse = ","))
}))))
paths$method <- factor(paths$method, levels = c("Newton", "Fisher scoring"))
p1 <- ggplot() +
  geom_raster(data = subset(g, notnd), aes(b0, b1), fill = PALETTE[["gold"]], alpha = 0.22) +
  geom_contour(data = subset(g, is.finite(dev)), aes(b0, b1, z = dev), breaks = c(2, 10, 30, 60, 100, 160, 240),
               colour = PALETTE[["grey"]], linewidth = 0.3) +
  geom_polygon(data = data.frame(b0 = c(0, 8, 0), b1 = c(0, -8, -8)), aes(b0, b1), fill = PALETTE[["light"]]) +
  geom_path(data = paths, aes(b0, b1, colour = method, group = interaction(method, start)), linewidth = 0.6) +
  geom_point(data = paths, aes(b0, b1, colour = method), size = 0.9) +
  geom_point(data = data.frame(b0 = c(6, 4), b1 = c(6, 0.5)), aes(b0, b1), colour = PALETTE[["ink"]], size = 2.2) +
  annotate("point", x = bhat[1], y = bhat[2], shape = 8, size = 2.5, colour = PALETTE[["ink"]]) +
  annotate("text", x = 4.75, y = 3.9, label = "observed information\nnot positive definite", hjust = 0, size = 2.8, colour = "#8a6418") +
  annotate("text", x = 6.9, y = 7.55, label = "Newton diverges", hjust = 1, size = 2.8, colour = PALETTE[["red"]]) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) +
  coord_cartesian(xlim = c(0, 8), ylim = c(-8, 8), expand = FALSE) +
  labs(x = expression(beta[0]), y = expression(beta[1]), title = "Gamma GLM, identity link, n = 60") +
  theme_blog()

errs <- do.call(rbind, lapply(c("newton", "fisher"), function(m) {
  P <- iterate(c(2, 0.3), m, 10)
  e <- sqrt(rowSums(sweep(P, 2, bhat)^2))
  data.frame(t = 0:10, err = e, method = ifelse(m == "newton", "Newton", "Fisher scoring"))
}))
errs <- subset(errs, err > 1e-14)
errs$method <- factor(errs$method, levels = c("Newton", "Fisher scoring"))
rho <- max(Mod(eigen(diag(2) - solve(exp_info(bhat), obs_info(bhat)))$values))
p2 <- ggplot(errs, aes(t, err, colour = method, shape = method)) +
  geom_line(data = data.frame(t = 0:10, err = errs$err[errs$method == "Fisher scoring"][2] * rho^((0:10) - 1)),
            aes(t, err), inherit.aes = FALSE, colour = PALETTE[["grey"]], linetype = "dotted") +
  geom_line(linewidth = 0.8) + geom_point(fill = "white", size = 1.8, stroke = 0.9) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) + scale_shape_manual(values = c(21, 22)) +
  coord_cartesian(ylim = c(1e-12, 10)) +
  labs(x = "iteration t", y = expression("||" * beta^(t) - hat(beta) * "||"),
       title = sprintf("From (2, 0.3); dotted: rho^t, rho = %.3f", rho)) +
  scale_x_continuous(breaks = seq(0, 10, 2)) +
  theme_blog() + theme(plot.title = element_text(size = 9.5))
save_fig(p1 + p2 + plot_layout(widths = c(1.1, 1)), here, "newton-vs-scoring", width = 7.9, height = 3.9)
