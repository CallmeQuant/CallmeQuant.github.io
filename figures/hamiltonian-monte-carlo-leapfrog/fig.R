# Leapfrog trajectories, energy error and integration time for HMC on Gaussian targets.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# Target N(0, Sigma), U(q) = q' Sigma^{-1} q / 2, identity mass matrix.
Sigma <- matrix(c(1, 0.95, 0.95, 1), 2)
Prec <- solve(Sigma)
sig_min <- sqrt(min(eigen(Sigma)$values))
Hfun <- function(q, p) 0.5 * rowSums((q %*% Prec) * q) + 0.5 * rowSums(p^2)

leapfrog <- function(q, p, eps, L) {           # q, p: 1 x d matrices; returns path
  qs <- q; ps <- p
  for (k in seq_len(L)) {
    p <- p - 0.5 * eps * q %*% Prec; q <- q + eps * p; p <- p - 0.5 * eps * q %*% Prec
    qs <- rbind(qs, q); ps <- rbind(ps, p)
  }
  list(q = qs, p = ps)
}

# ---- Figure 1
q0 <- matrix(c(-1.2, -0.9), 1); p0 <- matrix(c(1.4, -0.6), 1)
cases <- data.frame(eps = c(0.05, 0.40, 0.46), L = c(64, 8, 3))
cols <- unname(PALETTE[c("blue", "green", "red")])
labs_eps <- sprintf("epsilon == %s", c("0.05", "0.4", "0.46"))
paths <- do.call(rbind, lapply(1:3, function(i) {
  lf <- leapfrog(q0, p0, cases$eps[i], cases$L[i])
  data.frame(q1 = lf$q[, 1], q2 = lf$q[, 2], eps = labs_eps[i])
}))
errs <- do.call(rbind, lapply(1:3, function(i) {
  L <- round(6 / cases$eps[i]); lf <- leapfrog(q0, p0, cases$eps[i], L)
  data.frame(t = cases$eps[i] * (0:L), err = abs(Hfun(lf$q, lf$p) - Hfun(q0, p0)), eps = labs_eps[i])[-1, ]
}))
grid <- expand.grid(q1 = seq(-2.6, 2.6, length.out = 200), q2 = seq(-2.6, 2.6, length.out = 200))
grid$d <- exp(-Hfun(as.matrix(grid), matrix(0, nrow(grid), 2)))

pa <- ggplot() +
  geom_contour(data = grid, aes(q1, q2, z = d), breaks = c(0.05, 0.25, 0.5, 0.8),
               colour = PALETTE[["grey"]], linewidth = 0.35) +
  geom_path(data = paths, aes(q1, q2, colour = eps), linewidth = 0.45) +
  geom_point(data = paths, aes(q1, q2, colour = eps), size = 0.9) +
  annotate("point", x = -1.2, y = -0.9, shape = 15, size = 2.2, colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = cols, labels = scales::label_parse()) +
  coord_equal(xlim = c(-2.6, 2.6), ylim = c(-2.6, 2.6), expand = FALSE) +
  labs(x = expression(q[1]), y = expression(q[2]), title = "Leapfrog paths in position space") +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
pb <- ggplot(errs, aes(t, err, colour = eps)) + geom_line(linewidth = 0.5) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(1e-5, 1e4), xlim = c(0, 6)) +
  labs(x = expression("integration time" ~ k * epsilon), y = expression("|" * H(z[k]) - H(z[0]) * "|"),
       title = bquote("Energy error (stability limit" ~ 2 * sigma[min] == .(sprintf("%.3f", 2 * sig_min)) * ")")) +
  theme_blog()
save_fig(pa + pb + plot_layout(widths = c(1, 1.25)), here, "leapfrog-energy", width = 7.8, height = 3.7)

# ---- Figure 2: lag-one autocorrelation from one HMC transition started in stationarity
set.seed(11)
N <- 20000
one_step <- function(chol, eps, Tt) {
  d <- ncol(chol); P <- solve(chol %*% t(chol))
  q0 <- matrix(rnorm(N * d), N) %*% t(chol); p <- matrix(rnorm(N * d), N)
  L <- max(1, round(Tt / eps)); eps <- Tt / L; q <- q0
  h0 <- 0.5 * rowSums((q %*% P) * q) + 0.5 * rowSums(p^2)
  for (k in seq_len(L)) { p <- p - 0.5 * eps * q %*% P; q <- q + eps * p; p <- p - 0.5 * eps * q %*% P }
  h1 <- 0.5 * rowSums((q %*% P) * q) + 0.5 * rowSums(p^2)
  acc <- log(runif(N)) < h0 - h1
  q[!acc, ] <- q0[!acc, ]
  list(q0 = q0, q1 = q)
}

T1 <- seq(0, 2 * pi, length.out = 400); Te1 <- seq(0.15, 2 * pi, length.out = 24)
emp1 <- lapply(Te1, function(t) one_step(diag(1), 0.05, t))
th1 <- rbind(data.frame(T = T1, r = cos(T1), h = "h(q) == q * ':' ~ cos * T"),
             data.frame(T = T1, r = cos(T1)^2, h = "h(q) == q^2 * ':' ~ cos^2 * T"))
pts1 <- rbind(data.frame(T = Te1, r = sapply(emp1, function(e) cor(e$q0[, 1], e$q1[, 1])), h = th1$h[1]),
              data.frame(T = Te1, r = sapply(emp1, function(e) cor(e$q0[, 1]^2, e$q1[, 1]^2)), h = th1$h[401]))
p2a <- ggplot(th1, aes(T, r, colour = h)) + geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_line(linewidth = 0.7) + geom_point(data = pts1, size = 1.3) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), labels = scales::label_parse()) +
  scale_x_continuous(breaks = c(0, pi / 2, pi, 3 * pi / 2, 2 * pi),
                     labels = expression(0, pi / 2, pi, 3 * pi / 2, 2 * pi)) +
  coord_cartesian(ylim = c(-1.1, 1.1)) +
  labs(x = "integration time T", y = "lag-one autocorrelation", title = "N(0, 1)") +
  theme_blog() + theme(legend.direction = "vertical")

ev <- eigen(Sigma, symmetric = TRUE)            # decreasing: column 1 is the wide axis
sd <- sqrt(ev$values); V <- ev$vectors; chol <- t(chol(Sigma))
T2 <- seq(0, 2 * pi * sd[1], length.out = 600); Te2 <- seq(0.1, 2 * pi * sd[1], length.out = 26)
emp2 <- lapply(Te2, function(t) one_step(chol, 0.1 * sd[2], t))
nm <- sprintf(c("'wide axis,' ~ sigma == '%.2f'", "'narrow axis,' ~ sigma == '%.2f'"), sd)
th2 <- do.call(rbind, lapply(1:2, function(j) data.frame(T = T2, r = cos(T2 / sd[j]), ax = nm[j])))
pts2 <- do.call(rbind, lapply(1:2, function(j) data.frame(
  T = Te2, r = sapply(emp2, function(e) cor(e$q0 %*% V[, j], e$q1 %*% V[, j])), ax = nm[j])))
th2$ax <- factor(th2$ax, levels = nm); pts2$ax <- factor(pts2$ax, levels = nm)
p2b <- ggplot(th2, aes(T, r, colour = ax)) + geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_line(linewidth = 0.6) + geom_point(data = pts2, size = 1.3) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), labels = scales::label_parse()) +
  coord_cartesian(ylim = c(-1.1, 1.1)) +
  labs(x = "integration time T", y = NULL,
       title = expression(N(0, Sigma) * "," ~ rho == 0.95 * ": principal components")) +
  theme_blog() + theme(legend.direction = "vertical")
save_fig(p2a + p2b, here, "integration-time", width = 7.8, height = 3.8)
