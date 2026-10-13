# Conjugate gradients versus steepest descent: 2-d paths and A-norm error for n = 400.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

steepest <- function(A, b, x0, iters) {
  x <- x0; xs <- matrix(x0, nrow = 1)
  for (i in seq_len(iters)) { r <- drop(b - A %*% x); if (sum(r^2) == 0) break
    a <- sum(r^2) / drop(t(r) %*% A %*% r); x <- x + a * r; xs <- rbind(xs, x) }
  xs
}
cg <- function(A, b, x0, iters) {
  x <- x0; xs <- matrix(x0, nrow = 1); r <- drop(b - A %*% x); p <- r; rr <- sum(r^2)
  for (i in seq_len(iters)) { if (rr == 0) break
    Ap <- drop(A %*% p); a <- rr / sum(p * Ap); x <- x + a * p; r <- r - a * Ap
    rr_new <- sum(r^2); p <- r + (rr_new / rr) * p; rr <- rr_new; xs <- rbind(xs, x) }
  xs
}

# exact-arithmetic CG: minimize the A-norm error over x0 + K_k with a fully reorthogonalized Krylov basis
krylov_optimum <- function(A, b, x0, iters) {
  r0 <- drop(b - A %*% x0); V <- matrix(r0 / sqrt(sum(r0^2)), ncol = 1); xs <- matrix(x0, nrow = 1)
  for (k in seq_len(iters)) {
    y <- solve(t(V) %*% A %*% V, t(V) %*% r0); xs <- rbind(xs, x0 + drop(V %*% y))
    w <- drop(A %*% V[, ncol(V)])
    for (pass in 1:2) w <- w - drop(V %*% (t(V) %*% w))
    V <- cbind(V, w / sqrt(sum(w^2)))
  }
  xs
}

th <- 30 * pi / 180
R <- matrix(c(cos(th), sin(th), -sin(th), cos(th)), 2)
A2 <- R %*% diag(c(1, 10)) %*% t(R)
x0 <- drop(R %*% c(-2.7, 0.27))
sd <- steepest(A2, c(0, 0), x0, 25); cgp <- cg(A2, c(0, 0), x0, 2)
grid <- expand.grid(x1 = seq(-3, 3, length.out = 250), x2 = seq(-2.2, 2.2, length.out = 250))
G <- as.matrix(grid); grid$f <- 0.5 * rowSums((G %*% A2) * G)
paths <- rbind(data.frame(x1 = cgp[, 1], x2 = cgp[, 2], m = "conjugate gradients"),
               data.frame(x1 = sd[, 1], x2 = sd[, 2], m = "steepest descent"))
pa <- ggplot() +
  geom_contour(data = grid, aes(x1, x2, z = f), breaks = exp(seq(log(0.02), log(30), length.out = 10)),
               colour = PALETTE[["grey"]], linewidth = 0.3) +
  geom_path(data = subset(paths, m == "conjugate gradients"), aes(x1, x2, colour = m), linewidth = 1.6, alpha = 0.55) +
  geom_point(data = subset(paths, m == "conjugate gradients"), aes(x1, x2, colour = m), size = 2) +
  geom_path(data = subset(paths, m == "steepest descent"), aes(x1, x2, colour = m), linewidth = 0.45) +
  geom_point(data = subset(paths, m == "steepest descent"), aes(x1, x2, colour = m), size = 0.9) +
  annotate("point", x = 0, y = 0, shape = 8, size = 2.5) +
  scale_colour_manual(values = c("conjugate gradients" = PALETTE[["blue"]], "steepest descent" = PALETTE[["red"]])) +
  coord_equal(xlim = c(-3, 3), ylim = c(-2.2, 2.2), expand = FALSE) +
  labs(x = expression(x[1]), y = expression(x[2]), title = expression("(a)" ~ n == 2 * "," ~ kappa == 10)) +
  theme_blog() + theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0), legend.text = element_text(size = 7.5))

set.seed(5)
n <- 400; K <- 160; kappa <- 1000
Q <- qr.Q(qr(matrix(rnorm(n * n), n)))
lam_spread <- 10^seq(0, log10(kappa), length.out = n)
lam_clust <- c(runif(n - 5, 1, 2), c(200, 400, 600, 800, 1000))
xs_true <- rnorm(n)
anorm_err <- function(A, X) { E <- sweep(X, 2, xs_true); sqrt(rowSums((E %*% A) * E)) }
res <- list()
for (nm in c("spread", "clustered")) {
  lam <- if (nm == "spread") lam_spread else lam_clust
  A <- Q %*% (lam * t(Q)); b <- drop(A %*% xs_true)
  e_cg <- anorm_err(A, cg(A, b, rep(0, n), K)); e_sd <- anorm_err(A, steepest(A, b, rep(0, n), K))
  res[[nm]] <- list(cg = e_cg / e_cg[1], sd = e_sd / e_sd[1])
  if (nm == "clustered") { e_ex <- anorm_err(A, krylov_optimum(A, b, rep(0, n), 30)); res[[nm]]$exact <- e_ex / e_ex[1] }
}
pad <- function(v) c(v, rep(NA, K + 1 - length(v)))
k <- 0:K
db <- rbind(data.frame(k = k, v = pad(res$spread$sd), m = "steepest descent, spread"),
            data.frame(k = k, v = pad(res$spread$cg), m = "CG, spread spectrum"),
            data.frame(k = k, v = pad(res$clustered$cg), m = "CG, clustered + 5 outliers"),
            data.frame(k = k, v = pad(res$clustered$exact), m = "same, exact arithmetic"),
            data.frame(k = k, v = 2 * ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^k, m = "Chebyshev bound"))
db$m <- factor(db$m, levels = c("steepest descent, spread", "CG, spread spectrum", "CG, clustered + 5 outliers",
                                "same, exact arithmetic", "Chebyshev bound"))
pb <- ggplot(subset(db, is.na(v) | v > 1e-11), aes(k, v, colour = m, linetype = m)) + geom_line(linewidth = 0.8, na.rm = TRUE) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "green", "green", "blue")])) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed", "dotted")) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-10, 0, 2)) +
  coord_cartesian(ylim = c(1e-10, 3)) +
  labs(x = expression("iteration" ~ k), y = expression("||" * x[k] - x["*"] * "||"[A] / "||" * x[0] - x["*"] * "||"[A]),
       title = expression("(b)" ~ n == 400 * "," ~ kappa == 1000)) +
  theme_blog() + theme(legend.position = c(0.2, 0.02), legend.justification = c(0, 0), legend.text = element_text(size = 7.5), legend.key.height = unit(0.4, "cm"))
save_fig(pa + pb + plot_layout(widths = c(1, 1.25)), here, "cg-vs-steepest-descent", width = 7.8, height = 3.6)
