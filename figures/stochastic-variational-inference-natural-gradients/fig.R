# Stochastic variational inference for a Gaussian mixture with unknown means: paths of the variational
# means, and suboptimality of the ELBO against data processed for batch coordinate ascent and SVI.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(2013)
N <- 20000; K <- 4; d <- 2; SIG0 <- 10
centres <- rbind(c(-3, -1), c(0.5, 3), c(3, -2), c(-0.5, 0))
lab <- sample.int(K, N, replace = TRUE)
X <- centres[lab, ] + matrix(rnorm(N * d), N, d)
rowLSE <- function(A) { mx <- apply(A, 1, max); mx + log(rowSums(exp(A - mx))) }
exp_ll <- function(Xb, m, s2) {
  sq <- outer(rowSums(Xb^2), rep(1, K)) - 2 * Xb %*% t(m) + outer(rep(1, nrow(Xb)), rowSums(m^2))
  -0.5 * d * log(2 * pi) - 0.5 * (sq + d * outer(rep(1, nrow(Xb)), s2))
}
elbo_pp <- function(m, s2) {
  loc <- sum(rowLSE(log(1 / K) + exp_ll(X, m, s2)))
  pri <- sum(-0.5 * d * log(2 * pi * SIG0^2) - (rowSums(m^2) + d * s2) / (2 * SIG0^2))
  ent <- sum(0.5 * d * log(2 * pi * exp(1) * s2))
  (loc + pri + ent) / N
}
local_stats <- function(Xb, m, s2) {
  lp <- log(1 / K) + exp_ll(Xb, m, s2); phi <- exp(lp - rowLSE(lp))
  list(sx = t(phi) %*% Xb, sn = colSums(phi))
}
init_m <- matrix(rnorm(K * d, 0, 0.5), K, d); init_s2 <- rep(1, K)

m <- init_m; s2 <- init_s2; cavi <- numeric(40)
for (it in 1:40) {
  st <- local_stats(X, m, s2)
  eta2 <- -1 / (2 * SIG0^2) - 0.5 * st$sn; s2 <- -1 / (2 * eta2); m <- st$sx * s2
  cavi[it] <- elbo_pp(m, s2)
}
best <- max(cavi)

svi <- function(B, steps, kappa = 0.7, tau = 1, nrec = 60) {
  m <- init_m; s2 <- init_s2; e1 <- m / s2; e2 <- -1 / (2 * s2)
  rec <- unique(as.integer(exp(seq(0, log(steps), length.out = nrec))))
  out <- data.frame(); path <- list(m)
  for (t in 1:steps) {
    idx <- sample.int(N, B, replace = TRUE); st <- local_stats(X[idx, , drop = FALSE], m, s2)
    rho <- (t - 1 + tau)^(-kappa)
    e1 <- (1 - rho) * e1 + rho * (N / B) * st$sx
    e2 <- (1 - rho) * e2 + rho * (-1 / (2 * SIG0^2) - 0.5 * (N / B) * st$sn)
    s2 <- -1 / (2 * e2); m <- e1 * s2
    path[[t + 1]] <- m
    if (t %in% rec) out <- rbind(out, data.frame(pass = t * B / N, gap = max(best - elbo_pp(m, s2), 1e-9)))
  }
  list(out = out, path = path)
}
r10 <- svi(10, 12 * N / 10); r100 <- svi(100, 12 * N / 100)

P <- do.call(rbind, lapply(seq_along(r10$path), function(t) data.frame(step = t, k = factor(1:K), x1 = r10$path[[t]][, 1], x2 = r10$path[[t]][, 2])))
sub <- sample.int(N, 2500)
p1 <- ggplot() + geom_point(data = data.frame(x1 = X[sub, 1], x2 = X[sub, 2]), aes(x1, x2), colour = PALETTE[["grey"]], alpha = 0.3, size = 0.3) +
  geom_path(data = P, aes(x1, x2, colour = k), linewidth = 0.4) +
  geom_point(data = subset(P, step == 1), aes(x1, x2, colour = k), shape = 21, fill = "white", size = 1.8) +
  geom_point(data = subset(P, step == max(step)), aes(x1, x2, colour = k), size = 1.8) +
  geom_point(data = data.frame(x1 = centres[, 1], x2 = centres[, 2]), aes(x1, x2), shape = 4, size = 2.2, colour = PALETTE[["ink"]]) +
  scale_colour_blog(guide = "none") + coord_equal(xlim = c(-6.5, 6.5), ylim = c(-5.5, 6.5)) +
  labs(x = "x1", y = "x2", title = "Paths of the variational means (B = 10)") + theme_blog()
cg <- best - cavi[1:12]
D <- rbind(data.frame(pass = (1:12)[cg > 1e-9], gap = cg[cg > 1e-9], what = "batch coordinate ascent"),
           cbind(r10$out, what = "SVI, minibatch B = 10"), cbind(r100$out, what = "SVI, minibatch B = 100"))
D$what <- factor(D$what, levels = unique(D$what))
p2 <- ggplot(D, aes(pass, gap, colour = what)) + geom_line(linewidth = 0.6) +
  geom_point(data = subset(D, what == "batch coordinate ascent"), shape = 15, size = 1.4) +
  scale_x_log10() + scale_y_log10(limits = c(1e-7, 10)) +
  scale_colour_manual(values = c(PALETTE[["ink"]], CYCLE[1:2])) +
  guides(colour = guide_legend(ncol = 1)) +
  labs(x = "passes through the data", y = "(ELBO* - ELBO) / N", title = "Suboptimality against data processed") + theme_blog() +
  theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0))
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.25)), here, "svi-mixture", width = 8.2, height = 4.0)
