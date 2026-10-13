# Gaussian process regression: prior draws for three kernels, the posterior given noisy data, and the
# log marginal likelihood as a function of the length-scale.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(1996)
k_se <- function(a, b, ell, sf = 1) sf^2 * exp(-0.5 * outer(a, b, "-")^2 / ell^2)
k_exp <- function(a, b, ell, sf = 1) sf^2 * exp(-abs(outer(a, b, "-")) / ell)
xs <- seq(-3, 3, length.out = 400)

kern <- list("squared exponential, l = 1" = function(a, b) k_se(a, b, 1),
             "squared exponential, l = 0.25" = function(a, b) k_se(a, b, 0.25),
             "exponential, l = 1" = function(a, b) k_exp(a, b, 1))
d1 <- do.call(rbind, lapply(names(kern), function(nm) {
  L <- t(chol(kern[[nm]](xs, xs) + 1e-9 * diag(length(xs))))
  do.call(rbind, lapply(1:4, function(j) data.frame(x = xs, f = drop(L %*% rnorm(length(xs))), draw = factor(j), kernel = nm)))
}))
d1$kernel <- factor(d1$kernel, levels = names(kern))
p1 <- ggplot(d1, aes(x, f, colour = draw)) + annotate("rect", xmin = -3, xmax = 3, ymin = -2, ymax = 2, fill = PALETTE[["light"]]) +
  geom_line(linewidth = 0.45) + facet_wrap(~kernel) + scale_colour_blog(guide = "none") +
  coord_cartesian(xlim = c(-3, 3), ylim = c(-3.2, 3.2), expand = FALSE) + labs(x = "x", y = "f(x)") +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "prior-draws", width = 8.2, height = 2.9)

f_true <- function(x) sin(1.5 * x) + 0.3 * x
n <- 15; X <- sort(runif(n, -3, 3)); y <- f_true(X) + 0.25 * rnorm(n)
nlml_parts <- function(ell, sf, sn) {
  R <- chol(k_se(X, X, ell, sf) + sn^2 * diag(n)); a <- backsolve(R, forwardsolve(t(R), y))
  c(fit = 0.5 * sum(y * a), comp = sum(log(diag(R))), const = 0.5 * n * log(2 * pi))
}
nlml <- function(p) sum(nlml_parts(exp(p[1]), exp(p[2]), exp(p[3])))
fits <- lapply(list(log(c(0.5, 1, 0.3)), log(c(2, 1, 0.1)), log(c(0.2, 0.5, 0.5))), function(p0) optim(p0, nlml, method = "BFGS"))
best <- fits[[which.min(sapply(fits, `[[`, "value"))]]
hp <- exp(best$par); cat(sprintf("type-II ML: ell=%.3f sf=%.3f sigma=%.3f\n", hp[1], hp[2], hp[3]))
posterior <- function(xq, ell, sf, sn) {
  Ky <- k_se(X, X, ell, sf) + sn^2 * diag(n); Ks <- k_se(xq, X, ell, sf)
  list(mean = drop(Ks %*% solve(Ky, y)), cov = k_se(xq, xq, ell, sf) - Ks %*% solve(Ky, t(Ks)))
}
po <- posterior(xs, hp[1], hp[2], hp[3]); sd <- sqrt(pmax(diag(po$cov), 0))
Lp <- t(chol(po$cov + 1e-8 * diag(length(xs))))
dr <- do.call(rbind, lapply(1:3, function(j) data.frame(x = xs, f = po$mean + drop(Lp %*% rnorm(length(xs))), draw = j)))
p2 <- ggplot() +
  geom_ribbon(data = data.frame(x = xs, lo = po$mean - 1.96 * sd, hi = po$mean + 1.96 * sd), aes(x, ymin = lo, ymax = hi),
              fill = PALETTE[["blue"]], alpha = 0.15) +
  geom_line(data = dr, aes(x, f, group = draw), colour = PALETTE[["blue"]], linewidth = 0.3, alpha = 0.6) +
  geom_line(data = data.frame(x = xs, f = po$mean), aes(x, f), colour = PALETTE[["blue"]], linewidth = 0.9) +
  geom_line(data = data.frame(x = xs, f = f_true(xs)), aes(x, f), colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.5) +
  geom_point(data = data.frame(x = X, y = y), aes(x, y), colour = PALETTE[["red"]], size = 1.8) +
  coord_cartesian(xlim = c(-3, 3), ylim = c(-2.6, 2.6), expand = FALSE) +
  labs(x = "x", y = "f(x)", caption = "band: 95% pointwise; thick: posterior mean; thin: draws; dashed: true f") + theme_blog()
save_fig(p2, here, "posterior", width = 6.6, height = 3.5)

ells <- exp(seq(log(0.05), log(10), length.out = 300))
pp <- t(sapply(ells, nlml_parts, sf = hp[2], sn = hp[3]))
d3 <- rbind(data.frame(ell = ells, v = -rowSums(pp), what = "log p(y | l)"),
            data.frame(ell = ells, v = -pp[, "fit"], what = "-y'Ky^{-1}y/2 (fit)"),
            data.frame(ell = ells, v = -pp[, "comp"], what = "-log det Ky/2 (complexity)"))
d3$what <- factor(d3$what, levels = unique(d3$what))
p3 <- ggplot(d3, aes(ell, v, colour = what, linetype = what)) + geom_line(linewidth = 0.7) +
  geom_vline(xintercept = hp[1], colour = PALETTE[["grey"]], linetype = "dotted") +
  scale_x_log10() +
  scale_linetype_manual(values = c("solid", "dashed", "dashed"),
                        labels = expression(log ~ p(y ~ "|" ~ italic(l)), -y^T * K[y]^{-1} * y/2 ~ "(fit)",
                                            -log ~ det ~ K[y]/2 ~ "(complexity)")) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red", "blue")]),
                      labels = expression(log ~ p(y ~ "|" ~ italic(l)), -y^T * K[y]^{-1} * y/2 ~ "(fit)",
                                          -log ~ det ~ K[y]/2 ~ "(complexity)")) + guides(colour = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1)) +
  coord_cartesian(ylim = c(-40, 25)) + labs(x = expression("length-scale" ~ italic(l)), y = "nats",
       title = expression("Marginal likelihood (" * sigma[f] * "," ~ sigma ~ "fixed)")) + theme_blog()
d4 <- do.call(rbind, lapply(c(0.08, hp[1], 6), function(l) data.frame(x = xs, m = posterior(xs, l, hp[2], hp[3])$mean,
                                                                       lab = sprintf("%.2f", l))))
d4$lab <- factor(d4$lab, levels = unique(d4$lab))
p4 <- ggplot(d4, aes(x, m, colour = lab)) + geom_line(linewidth = 0.7) +
  geom_point(data = data.frame(x = X, m = y), aes(x, m), inherit.aes = FALSE, colour = PALETTE[["red"]], size = 1.5) +
  scale_colour_manual(values = CYCLE[c(4, 1, 3)], labels = function(b) parse(text = paste("italic(l) ==", b))) + coord_cartesian(xlim = c(-3, 3), ylim = c(-2.6, 2.6)) +
  labs(x = "x", y = NULL, title = "Posterior means") + theme_blog()
save_fig(p3 + p4, here, "marginal-likelihood", width = 8.2, height = 3.8)
