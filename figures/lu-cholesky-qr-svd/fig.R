# Accuracy of least squares via the normal equations (Cholesky), Householder QR and the SVD.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

set.seed(31)
n <- 100; p <- 10; reps <- 20
kappas <- 10^seq(0, 9, length.out = 19)
u <- .Machine$double.eps / 2

solve_normal <- function(A, y) tryCatch({
  R <- chol(crossprod(A))                               # A^T A = R^T R
  backsolve(R, forwardsolve(t(R), crossprod(A, y)))
}, error = function(e) rep(NA_real_, ncol(A)))
solve_qr <- function(A, y) qr.coef(qr(A, LAPACK = TRUE), y)   # Householder QR (LAPACK)
solve_svd <- function(A, y) { s <- svd(A); s$v %*% (crossprod(s$u, y) / s$d) }

methods <- list("normal equations + Cholesky" = solve_normal, "Householder QR" = solve_qr, "SVD" = solve_svd)
res <- do.call(rbind, lapply(seq_along(kappas), function(i) {
  e <- sapply(seq_len(reps), function(r) {
    U <- qr.Q(qr(matrix(rnorm(n * p), n, p)))
    V <- qr.Q(qr(matrix(rnorm(p * p), p, p)))
    A <- U %*% diag(10^seq(0, -log10(kappas[i]), length.out = p)) %*% t(V)
    beta <- rnorm(p); beta <- beta / sqrt(sum(beta^2))
    y <- A %*% beta
    sapply(methods, function(f) { b <- f(A, y); if (any(!is.finite(b))) NA else sqrt(sum((b - beta)^2)) })
  })
  med <- apply(e, 1, function(v) if (any(is.na(v))) NA else median(v))
  data.frame(kappa = kappas[i], method = names(methods), err = pmin(med, 1))
}))
res$method <- factor(res$method, levels = names(methods))
ref <- data.frame(kappa = kappas, k1 = u * kappas, k2 = u * kappas^2)

p1 <- ggplot(res, aes(kappa, err, colour = method, shape = method)) +
  geom_line(data = ref, aes(kappa, k1), inherit.aes = FALSE, colour = PALETTE[["grey"]], linetype = "dashed", linewidth = 0.4) +
  geom_line(data = subset(ref, k2 < 3), aes(kappa, k2), inherit.aes = FALSE, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.5) +
  annotate("text", x = 1e6, y = u * 1e6 * 5, label = "u * kappa", parse = TRUE, hjust = 1, vjust = 0, colour = PALETTE[["grey"]]) +
  annotate("text", x = 3e5, y = u * 9e10 * 3, label = "u * kappa^2", parse = TRUE, hjust = 1, vjust = 0, colour = PALETTE[["grey"]]) +
  geom_line(linewidth = 0.8, na.rm = TRUE) + geom_point(size = 1.8, na.rm = TRUE) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "green")])) +
  scale_shape_manual(values = c(16, 15, 17)) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(0, 8, 2)) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-16, 0, 2)) +
  coord_cartesian(ylim = c(1e-17, 3)) +
  labs(x = expression("condition number" ~ kappa[2](A) == sigma[1] / sigma[p]),
       y = expression("relative error of" ~ hat(beta))) +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
save_fig(p1, here, "least-squares-accuracy", width = 6.4, height = 3.9)
