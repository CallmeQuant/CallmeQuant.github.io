# Trace-ratio objective maximized by Riemannian gradient ascent on the Stiefel manifold,
# compared with PCA and with the generalized-eigenvector (ratio-trace) solution.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(0)
d <- 6; r <- 2; nper <- 100
A <- matrix(rnorm(d * d), d) %*% diag(c(3, 2.5, 2, 1, 0.6, 0.4))
means <- matrix(rnorm(3 * d), 3)
X <- do.call(rbind, lapply(1:3, function(c) matrix(rnorm(nper * d), nper) %*% t(A) + matrix(means[c, ], nper, d, byrow = TRUE)))
lab <- rep(1:3, each = nper)
mu <- colMeans(X)
Sw <- Reduce(`+`, lapply(1:3, function(c) { Z <- scale(X[lab == c, ], scale = FALSE); crossprod(Z) })) / nrow(X)
Sb <- Reduce(`+`, lapply(1:3, function(c) mean(lab == c) * tcrossprod(colMeans(X[lab == c, ]) - mu)))
St <- crossprod(sweep(X, 2, mu)) / nrow(X)
f <- function(M) sum(diag(t(M) %*% Sb %*% M)) / sum(diag(t(M) %*% Sw %*% M))
qf <- function(B) { q <- qr(B); Q <- qr.Q(q); Q %*% diag(sign(diag(qr.R(q)))) }
tangent <- function(M, G) { S <- t(M) %*% G; G - M %*% (S + t(S)) / 2 }
egrad <- function(M) { a <- sum(diag(t(M) %*% Sb %*% M)); b <- sum(diag(t(M) %*% Sw %*% M)); 2 * Sb %*% M / b - 2 * a / b^2 * Sw %*% M }
ascend <- function(M, iters) {
  vals <- f(M)
  for (k in seq_len(iters)) {
    Z <- tangent(M, egrad(M)); t <- 1
    while (f(qf(M + t * Z)) < f(M) + 1e-4 * t * sum(Z * Z) && t > 1e-10) t <- t / 2
    M <- qf(M + t * Z); vals <- c(vals, f(M))
  }
  list(M = M, vals = vals)
}
M_pca <- eigen(St, symmetric = TRUE)$vectors[, 1:r]
L <- t(chol(Sw)); Li <- solve(L)
M_gev <- qf((t(Li) %*% eigen(Li %*% Sb %*% t(Li), symmetric = TRUE)$vectors)[, 1:r])
runs <- lapply(1:5, function(i) ascend(qf(matrix(rnorm(d * r), d)), 600))
M_opt <- runs[[which.max(sapply(runs, function(o) tail(o$vals, 1)))]]$M
rho_star <- uniroot(function(rho) sum(eigen(Sb - rho * Sw, symmetric = TRUE, only.values = TRUE)$values[1:r]),
                    c(0, 50), tol = 1e-12)$root

proj_plot <- function(M, title) {
  Y <- sweep(X, 2, mu) %*% M
  ggplot(data.frame(y1 = Y[, 1], y2 = Y[, 2], c = factor(lab)), aes(y1, y2, colour = c)) +
    geom_point(size = 0.7, alpha = 0.7) + scale_colour_manual(values = CYCLE[1:3], guide = "none") +
    labs(x = expression(y[1]), y = expression(y[2]), title = sprintf("%s, f(M) = %.2f", title, f(M))) +
    theme_blog() + theme(plot.title = element_text(size = 9))
}
dv <- do.call(rbind, lapply(1:5, function(i) data.frame(k = seq_along(runs[[i]]$vals) - 1, v = runs[[i]]$vals, run = i)))
hl <- data.frame(y = c(rho_star, f(M_gev), f(M_pca)), w = c("global maximum", "ratio-trace optimum", "PCA"))
hl$w <- factor(hl$w, levels = hl$w)
pd <- ggplot() +
  geom_hline(data = hl, aes(yintercept = y, linetype = w), colour = c(PALETTE[["ink"]], PALETTE[["gold"]], PALETTE[["grey"]])) +
  geom_line(data = dv, aes(k + 1, v, group = run), colour = PALETTE[["blue"]], linewidth = 0.5) +
  scale_linetype_manual(values = c("solid", "dashed", "dotted")) +
  scale_x_log10() +
  labs(x = "iteration k + 1 (log scale)", y = expression(f(M[k])), title = "(d) Objective along the\niterations") +
  theme_blog() + theme(legend.position = "bottom", legend.direction = "vertical", plot.title = element_text(size = 9),
                       legend.text = element_text(size = 7))
p <- (proj_plot(M_pca, "(a) PCA\n(top eigenvectors of the covariance)") + proj_plot(M_gev, "(b) ratio-trace optimum\n(generalized eigenvectors)")) /
  (proj_plot(M_opt, "(c) trace-ratio optimum\n(Riemannian ascent)") + pd)
save_fig(p, here, "trace-ratio-projections", width = 7.2, height = 6.6)
