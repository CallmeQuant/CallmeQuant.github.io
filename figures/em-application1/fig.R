# EM for a three-component bivariate Gaussian mixture.
# gmm-data: complete data coloured by component versus the observed data.
# em-iterations: fitted components at several EM iterations and the log-likelihood trace.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(MASS); library(mvtnorm) })

set.seed(91)
w_true <- c(0.4, 0.3, 0.3)
mu_true <- list(c(-3, 0), c(2, 2.5), c(2.5, -2))
cov_true <- list(matrix(c(1.2, 0.6, 0.6, 1.0), 2), matrix(c(0.8, -0.3, -0.3, 0.6), 2),
                 matrix(c(1.5, 0, 0, 0.4), 2))
n <- 450
z <- sample(1:3, n, replace = TRUE, prob = w_true)
X <- t(sapply(z, function(k) mvrnorm(1, mu_true[[k]], cov_true[[k]])))
cols <- unname(PALETTE[c("blue", "red", "green")])

d <- data.frame(x1 = X[, 1], x2 = X[, 2], z = factor(z))
pa <- ggplot(d, aes(x1, x2, colour = z)) + geom_point(size = 0.9, alpha = 0.75) +
  scale_colour_manual(values = cols, labels = c("z = 1", "z = 2", "z = 3")) +
  coord_equal() + labs(x = expression(x[i1]), y = expression(x[i2]), title = "(a) complete data") +
  theme_blog() + theme(legend.position = "bottom")
pb <- ggplot(d, aes(x1, x2)) + geom_point(size = 0.9, alpha = 0.75, colour = PALETTE[["grey"]]) +
  coord_equal() + labs(x = expression(x[i1]), y = NULL, title = "(b) observed data") + theme_blog()
save_fig(pa + pb, here, "gmm-data", width = 7.2, height = 3.6)

e_step <- function(w, mu, S) {
  logp <- sapply(1:3, function(k) log(w[k]) + dmvnorm(X, mu[[k]], S[[k]], log = TRUE))
  m <- apply(logp, 1, max)
  ll <- m + log(rowSums(exp(logp - m)))
  list(g = exp(logp - ll), ll = sum(ll))
}
m_step <- function(g) {
  Nk <- colSums(g)
  mu <- lapply(1:3, function(k) colSums(g[, k] * X) / Nk[k])
  S <- lapply(1:3, function(k) { D <- sweep(X, 2, mu[[k]]); crossprod(D * g[, k], D) / Nk[k] })
  list(w = Nk / n, mu = mu, S = S)
}
ellipse <- function(mu, S, level = 5.991) {
  e <- eigen(S, symmetric = TRUE); th <- seq(0, 2 * pi, length.out = 200)
  circ <- cbind(cos(th), sin(th)) %*% diag(sqrt(level * e$values))
  sweep(circ %*% t(e$vectors), 2, mu, "+")
}

init <- c(which(z == 1)[1:2], which(z == 3)[1])
par <- list(w = rep(1 / 3, 3), mu = lapply(init, function(i) X[i, ]), S = rep(list(cov(X)), 3))
snaps <- list(`0` = par); lls <- c()
for (it in 1:200) {
  es <- e_step(par$w, par$mu, par$S)
  lls <- c(lls, es$ll)
  par <- m_step(es$g)
  snaps[[as.character(it)]] <- par
  if (it > 1 && lls[it] - lls[it - 1] < 1e-8) break
}
show <- c(0, 1, 4, 10, it)
rgbm <- t(col2rgb(cols)) / 255
panels <- lapply(show, function(t) {
  p <- snaps[[as.character(t)]]
  g <- e_step(p$w, p$mu, p$S)$g
  mix <- g %*% rgbm
  dd <- data.frame(x1 = X[, 1], x2 = X[, 2], col = rgb(mix[, 1], mix[, 2], mix[, 3]))
  el <- do.call(rbind, lapply(1:3, function(k) data.frame(ellipse(p$mu[[k]], p$S[[k]]), k = k)))
  names(el)[1:2] <- c("x1", "x2")
  cen <- data.frame(x1 = sapply(p$mu, `[`, 1), x2 = sapply(p$mu, `[`, 2))
  ggplot() + geom_point(data = dd, aes(x1, x2, colour = I(col)), size = 0.5, alpha = 0.6) +
    geom_path(data = el, aes(x1, x2, group = k), colour = PALETTE[["ink"]], linewidth = 0.45) +
    geom_point(data = cen, aes(x1, x2), shape = 4, size = 2, stroke = 1, colour = PALETTE[["ink"]]) +
    coord_equal(xlim = c(-7, 7), ylim = c(-6, 6), expand = FALSE) +
    labs(x = NULL, y = NULL, title = paste("iteration", t)) + theme_blog(base_size = 9)
})
pl <- ggplot(data.frame(t = seq_along(lls) - 1, ll = lls), aes(t, ll)) +
  geom_line(colour = PALETTE[["blue"]], linewidth = 0.5) + geom_point(colour = PALETTE[["blue"]], size = 0.8) +
  labs(x = "iteration t", y = NULL, title = "log-likelihood") + theme_blog(base_size = 9)
save_fig(wrap_plots(c(panels, list(pl)), ncol = 3), here, "em-iterations", width = 8, height = 5.4)
cat("iterations", it, "\n")
