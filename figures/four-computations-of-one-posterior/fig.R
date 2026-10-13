# One posterior, four computations: a two-parameter logistic regression (n = 25) computed exactly on
# a grid, by the Laplace approximation, by random-walk Metropolis and by mean-field Gaussian
# variational inference.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1986)
n <- 25; prior_sd <- 2.5
x <- runif(n, 0, 3)
y <- as.numeric(runif(n) < plogis(-2 + 1.5 * x))
loglik <- function(a, b) sum(y * plogis(a + b * x, log.p = TRUE) + (1 - y) * plogis(-(a + b * x), log.p = TRUE))
logpost <- function(a, b) loglik(a, b) - (a^2 + b^2) / (2 * prior_sd^2)
lp <- function(t) logpost(t[1], t[2])

mle <- optim(c(0, 0), function(t) -loglik(t[1], t[2]), method = "BFGS")$par
mode <- optim(c(0, 0), function(t) -lp(t), method = "BFGS")$par
lap_cov <- solve(optimHess(mode, function(t) -lp(t)))

A <- seq(-5.5, 3, length.out = 501); B <- seq(-1.5, 3.5, length.out = 501)
dA <- A[2] - A[1]; dB <- B[2] - B[1]
G <- expand.grid(a = A, b = B)
LP <- mapply(logpost, G$a, G$b)
G$p <- exp(LP - max(LP)); G$p <- G$p / (sum(G$p) * dA * dB)
post_mean <- c(sum(G$a * G$p), sum(G$b * G$p)) * dA * dB

iters <- 20000; burn <- 2000
L <- t(chol(2.38^2 / 2 * lap_cov))
cur <- mode; cur_lp <- lp(cur); chain <- matrix(NA, iters, 2)
for (t in 1:iters) {
  prop <- cur + drop(L %*% rnorm(2)); prop_lp <- lp(prop)
  if (log(runif(1)) < prop_lp - cur_lp) { cur <- prop; cur_lp <- prop_lp }
  chain[t, ] <- cur
}
chain <- chain[-(1:burn), ]

# probabilists' Gauss-Hermite nodes by the Golub-Welsch algorithm
gh <- function(m) {
  J <- matrix(0, m, m); J[cbind(1:(m - 1), 2:m)] <- sqrt(1:(m - 1)); J <- J + t(J)
  e <- eigen(J, symmetric = TRUE); list(x = e$values, w = e$vectors[1, ]^2)
}
q <- gh(30)
neg_elbo <- function(par) {
  a <- par[1] + exp(par[3]) * q$x; b <- par[2] + exp(par[4]) * q$x
  E <- sum(outer(q$w, q$w) * outer(a, b, Vectorize(logpost)))
  -(E + par[3] + par[4])
}
vi <- optim(c(mode, 0, 0), neg_elbo, method = "Nelder-Mead", control = list(reltol = 1e-12, maxit = 5000))$par
vi_m <- vi[1:2]; vi_s <- exp(vi[3:4])

gauss <- function(m, C) {
  Ci <- solve(C); da <- G$a - m[1]; db <- G$b - m[2]
  exp(-(Ci[1, 1] * da^2 + 2 * Ci[1, 2] * da * db + Ci[2, 2] * db^2) / 2) / (2 * pi * sqrt(det(C)))
}
levels_of <- function(d) { s <- sort(d, decreasing = TRUE); c <- cumsum(s) * dA * dB
  sort(sapply(c(0.99, 0.9, 0.5), function(qq) s[which(c >= qq)[1]])) }
G$lap <- gauss(mode, lap_cov); G$vi <- gauss(vi_m, diag(vi_s^2))

base <- function(title, col = PALETTE[["grey"]]) ggplot(G, aes(a, b)) +
  geom_contour(aes(z = p), breaks = levels_of(G$p), colour = col, linewidth = 0.45) +
  coord_cartesian(xlim = c(-5.2, 1.8), ylim = c(-1, 3.2), expand = FALSE) +
  labs(x = expression("intercept" ~ alpha), y = expression("slope" ~ beta), title = title) + theme_blog()
pts <- data.frame(a = c(mle[1], mode[1], post_mean[1]), b = c(mle[2], mode[2], post_mean[2]),
                  what = factor(c("MLE", "MAP", "posterior mean"), levels = c("MLE", "MAP", "posterior mean")))
p1 <- base("Exact (grid)", PALETTE[["ink"]]) +
  geom_point(data = pts, aes(a, b, colour = what, shape = what), size = 2) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "gold")])) + scale_shape_manual(values = c(15, 16, 17)) +
  theme(legend.position = c(0.25, 0.15), legend.justification = c(0.5, 0.5))
p2 <- base("Laplace approximation") +
  geom_contour(aes(z = lap), breaks = levels_of(G$lap), colour = PALETTE[["red"]], linetype = "dashed", linewidth = 0.55)
idx <- sample(nrow(chain), 1500)
p3 <- base("Random-walk Metropolis") +
  geom_point(data = data.frame(a = chain[idx, 1], b = chain[idx, 2]), aes(a, b), colour = PALETTE[["green"]],
             size = 0.35, alpha = 0.5)
p4 <- base("Mean-field variational") +
  geom_contour(aes(z = vi), breaks = levels_of(G$vi), colour = PALETTE[["gold"]], linetype = "dashed", linewidth = 0.55)
save_fig((p1 + p2) / (p3 + p4), here, "four-posteriors", width = 7.2, height = 6.6)

pb <- tapply(G$p, G$b, sum) * dA
d5 <- rbind(data.frame(b = B, d = as.numeric(pb), what = "exact"),
            data.frame(b = B, d = dnorm(B, mode[2], sqrt(lap_cov[2, 2])), what = "Laplace"),
            data.frame(b = B, d = dnorm(B, vi_m[2], vi_s[2]), what = "mean-field VI"))
d5$what <- factor(d5$what, levels = c("exact", "Laplace", "mean-field VI"))
p5 <- ggplot() +
  geom_histogram(data = data.frame(b = chain[, 2]), aes(b, after_stat(density)), bins = 70, boundary = -0.6,
                 fill = PALETTE[["light"]], colour = "white", linewidth = 0.2) +
  geom_line(data = d5, aes(b, d, colour = what, linetype = what), linewidth = 0.7) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red", "violet")])) +
  scale_linetype_manual(values = c("solid", "dashed", "dotted")) +
  coord_cartesian(xlim = c(-0.6, 2.6), ylim = c(0, 1.12 * max(d5$d)), expand = FALSE) +
  labs(x = expression("slope" ~ beta), y = "marginal posterior density") +
  theme_blog() + theme(legend.position = c(0.85, 0.8), legend.justification = c(0.5, 0.5))
save_fig(p5, here, "slope-marginal", width = 6.4, height = 3.2)
