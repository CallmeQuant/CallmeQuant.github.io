# BFGS on the Rosenbrock function versus gradient descent and Newton's method, and the
# hereditary property of BFGS on a quadratic.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

f <- function(x) (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
grad <- function(x) c(-2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2), 200 * (x[2] - x[1]^2))
hess <- function(x) matrix(c(2 - 400 * x[2] + 1200 * x[1]^2, -400 * x[1], -400 * x[1], 200), 2)
wolfe <- function(x, p, g, c1 = 1e-4, c2 = 0.9, amax = 50) {
  lo <- 0; hi <- Inf; a <- 1; f0 <- f(x); d0 <- sum(g * p)
  for (i in 1:60) {
    if (f(x + a * p) > f0 + c1 * a * d0) hi <- a
    else if (sum(grad(x + a * p) * p) < c2 * d0) lo <- a
    else return(a)
    a <- if (is.finite(hi)) (lo + hi) / 2 else min(2 * a, amax)
  }
  a
}
backtrack <- function(x, p, g, c1 = 1e-4) { a <- 1; while (f(x + a * p) > f(x) + c1 * a * sum(g * p)) a <- a / 2; a }
run <- function(method, x0, iters, tol = 1e-10) {
  x <- x0; H <- diag(2); xs <- matrix(x, 1); gs <- sqrt(sum(grad(x)^2))
  for (i in seq_len(iters)) {
    g <- grad(x); if (sqrt(sum(g^2)) < tol) break
    if (method == "gd") { p <- -g; a <- 1e-3 }
    else if (method == "newton") { p <- -solve(hess(x), g); a <- backtrack(x, p, g) }
    else { p <- -drop(H %*% g); a <- wolfe(x, p, g) }
    xn <- x + a * p
    if (method == "bfgs") {
      s <- xn - x; y <- grad(xn) - g; rho <- 1 / sum(y * s)
      if (nrow(xs) == 1) H <- sum(y * s) / sum(y * y) * diag(2)
      V <- diag(2) - rho * s %*% t(y)
      H <- V %*% H %*% t(V) + rho * s %*% t(s)
    }
    x <- xn; xs <- rbind(xs, x); gs <- c(gs, sqrt(sum(grad(x)^2)))
  }
  list(xs = xs, gs = gs)
}
x0 <- c(-1.2, 1)
runs <- list("gradient descent" = run("gd", x0, 3000), "BFGS" = run("bfgs", x0, 200), "Newton" = run("newton", x0, 200))
lab <- sapply(names(runs), function(nm) sprintf("%s (%d it.)", nm, nrow(runs[[nm]]$xs) - 1))
cols <- setNames(unname(PALETTE[c("red", "blue", "green")]), lab)
paths <- do.call(rbind, lapply(names(runs), function(nm) data.frame(x1 = runs[[nm]]$xs[, 1], x2 = runs[[nm]]$xs[, 2], m = lab[[nm]])))
paths$m <- factor(paths$m, levels = lab)
grid <- expand.grid(x1 = seq(-1.6, 1.5, length.out = 300), x2 = seq(-0.6, 1.6, length.out = 300))
grid$f <- (1 - grid$x1)^2 + 100 * (grid$x2 - grid$x1^2)^2
pa <- ggplot() +
  geom_contour(data = grid, aes(x1, x2, z = f), breaks = exp(seq(log(0.05), log(800), length.out = 14)),
               colour = PALETTE[["grey"]], linewidth = 0.25) +
  geom_path(data = paths, aes(x1, x2, colour = m), linewidth = 0.6) +
  geom_point(data = subset(paths, m != lab[["gradient descent"]]), aes(x1, x2, colour = m), size = 0.9) +
  annotate("point", x = 1, y = 1, shape = 8, size = 2.5) + annotate("point", x = -1.2, y = 1, shape = 15, size = 2) +
  annotate("point", x = tail(runs[["gradient descent"]]$xs[, 1], 1), y = tail(runs[["gradient descent"]]$xs[, 2], 1),
           colour = PALETTE[["red"]], size = 1.8) +
  scale_colour_manual(values = cols) +
  coord_cartesian(xlim = c(-1.6, 1.5), ylim = c(-0.6, 1.6), expand = FALSE) +
  labs(x = expression(x[1]), y = expression(x[2]), title = "(a) Rosenbrock function, start (-1.2, 1)") +
  theme_blog() + theme(legend.position = "bottom", legend.direction = "horizontal", legend.text = element_text(size = 7.5)) + guides(colour = guide_legend(nrow = 2))
gnorm <- do.call(rbind, lapply(names(runs), function(nm) data.frame(k = seq_along(runs[[nm]]$gs) - 1, g = runs[[nm]]$gs, m = nm)))
gnorm$m <- factor(gnorm$m, levels = names(runs))
gd_final <- tail(runs[["gradient descent"]]$gs, 1)
pb <- ggplot(subset(gnorm, k <= 60), aes(k, g, colour = m)) + geom_line(linewidth = 0.7) +
  annotate("text", x = 58, y = 4e-3, hjust = 1, vjust = 1, colour = PALETTE[["red"]], size = 3,
           label = sprintf("gradient descent:\n%.2f after 3000 it.", gd_final)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "green")]), guide = "none") +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-10, 4, 2)) +
  coord_cartesian(xlim = c(0, 60), ylim = c(1e-11, 1e4)) +
  labs(x = expression("iteration" ~ k), y = expression("||" * nabla * f(x[k]) * "||"), title = "(b) Gradient norm") +
  theme_blog()
save_fig(pa + pb + plot_layout(widths = c(1.1, 1)), here, "rosenbrock", width = 7.8, height = 3.9)

# ---- Figure 2 ------------------------------------------------------------------------------
set.seed(8)
n <- 8
Q <- qr.Q(qr(matrix(rnorm(n * n), n)))
A <- Q %*% (10^seq(0.3, 2.3, length.out = n) * t(Q))
b <- rnorm(n); x <- rep(0, n); H <- diag(n)
eig <- function(H) sort(Re(eigen(H %*% A, only.values = TRUE)$values))
E <- list(eig(H))
for (k in 1:n) {
  g <- drop(A %*% x - b); p <- -drop(H %*% g); a <- -sum(g * p) / drop(t(p) %*% A %*% p)
  s <- a * p; y <- drop(A %*% s); rho <- 1 / sum(y * s); V <- diag(n) - rho * s %*% t(y)
  H <- V %*% H %*% t(V) + rho * s %*% t(s); x <- x + s
  E[[k + 1]] <- eig(H)
}
de <- do.call(rbind, lapply(0:n, function(k) data.frame(k = k, e = E[[k + 1]], one = abs(E[[k + 1]] - 1) < 1e-6)))
cnt <- aggregate(one ~ k, de, sum)
p2 <- ggplot(de, aes(k, e, colour = one)) +
  geom_hline(yintercept = 1, colour = PALETTE[["blue"]], linetype = "dotted", linewidth = 0.4) +
  geom_point(size = 2) +
  geom_text(data = cnt, aes(k, 400, label = one), inherit.aes = FALSE, colour = PALETTE[["blue"]], size = 3.2) +
  scale_colour_manual(values = c(`TRUE` = PALETTE[["blue"]], `FALSE` = PALETTE[["grey"]]), guide = "none") +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_x_continuous(breaks = 0:n) + coord_cartesian(ylim = c(0.5, 900)) +
  labs(x = expression("iteration" ~ k), y = expression("eigenvalues of" ~ H[k] * A),
       title = "Quadratic, n = 8, exact line search; top: number of eigenvalues equal to 1") +
  theme_blog()
save_fig(p2, here, "hereditary-eigenvalues", width = 6.0, height = 3.3)
