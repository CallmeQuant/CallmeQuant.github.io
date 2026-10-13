# Kernel ridge regression: the fitted function as a sum of kernel sections at the data, and choice of the
# penalty by the closed-form leave-one-out formula.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(1971)
WIDTH <- 0.08
kern <- function(a, b) exp(-0.5 * outer(a, b, "-")^2 / WIDTH^2)
f_true <- function(x) x * sin(3 * pi * x) + 0.3
n <- 40; X <- sort(runif(n)); y <- f_true(X) + 0.2 * rnorm(n)
K <- kern(X, X); xs <- seq(0, 1, length.out = 500); Ks <- kern(xs, X)
fit <- function(lam) solve(K + n * lam * diag(n), y)
loo <- function(lam) { H <- K %*% solve(K + n * lam * diag(n)); r <- (y - H %*% y) / (1 - diag(H)); c(cv = mean(r^2), df = sum(diag(H))) }
loo_brute <- function(lam) mean(sapply(1:n, function(i) {
  a <- solve(K[-i, -i] + n * lam * diag(n - 1), y[-i]); y[i] - sum(K[i, -i] * a) })^2)
lams <- exp(seq(log(1e-6), log(1), length.out = 120))
cv <- t(sapply(lams, loo)); lam_hat <- lams[which.min(cv[, "cv"])]
alpha <- fit(lam_hat)

d1 <- rbind(data.frame(x = xs, v = f_true(xs), what = "true regression function"),
            data.frame(x = xs, v = drop(Ks %*% alpha), what = "fit: sum_i alpha_i k(., x_i)"))
p1 <- ggplot(d1, aes(x, v, colour = what, linetype = what)) + geom_line(linewidth = 0.8) +
  geom_point(data = data.frame(x = X, v = y), aes(x, v), inherit.aes = FALSE, colour = PALETTE[["red"]], size = 1.5) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "ink")]), labels = expression(hat(f) == sum(hat(alpha)[i] ~ k(x, x[i]), i), "true regression function")) + scale_linetype_manual(values = c("solid", "dashed"), labels = expression(hat(f) == sum(hat(alpha)[i] ~ k(x, x[i]), i), "true regression function")) +
  coord_cartesian(ylim = c(-1.3, 1.6)) + labs(x = NULL, y = "y") + theme_blog()
d2 <- do.call(rbind, lapply(1:n, function(i) data.frame(x = xs, v = alpha[i] * Ks[, i], i = i, col = factor(i %% 4))))
p2 <- ggplot(d2, aes(x, v, group = i, colour = col)) + geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_segment(data = data.frame(x = X, v = alpha), aes(x = x, xend = x, y = 0, yend = v), inherit.aes = FALSE,
               colour = PALETTE[["grey"]], linewidth = 0.3) +
  geom_line(linewidth = 0.4, alpha = 0.85) + scale_colour_blog(guide = "none") +
  labs(x = "x", y = expression(hat(alpha)[i] ~ k(x, x[i]))) + theme_blog()
save_fig(p1 / p2 + plot_layout(heights = c(1.3, 1)), here, "kernel-sections", width = 6.6, height = 4.8)

d3 <- do.call(rbind, lapply(c(1e-6, lam_hat, 0.3), function(l) data.frame(x = xs, v = drop(Ks %*% fit(l)), lab = sprintf("lambda = %.2g", l))))
d3$lab <- factor(d3$lab, levels = unique(d3$lab))
p3 <- ggplot(d3, aes(x, v, colour = lab)) + geom_line(linewidth = 0.7) +
  geom_point(data = data.frame(x = X, v = y), aes(x, v), inherit.aes = FALSE, colour = PALETTE[["red"]], size = 1.3) +
  scale_colour_manual(values = CYCLE[c(4, 1, 3)], labels = c(expression(lambda == 10^-6), bquote(hat(lambda) == .(sprintf("%.4f", lam_hat))), expression(lambda == 0.3))) + coord_cartesian(xlim = c(0, 1), ylim = c(-1.3, 1.6)) +
  labs(x = "x", y = NULL, colour = NULL, title = "Fits for three penalties") + theme_blog() + theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0))
chk <- lams[seq(1, 120, by = 12)]
p4 <- ggplot(data.frame(l = lams, cv = cv[, "cv"]), aes(l, cv)) + geom_line(colour = PALETTE[["blue"]], linewidth = 0.8) +
  geom_point(data = data.frame(l = chk, cv = sapply(chk, loo_brute)), shape = 21, fill = "white", colour = PALETTE[["red"]], size = 2) +
  geom_vline(xintercept = lam_hat, linetype = "dotted", colour = PALETTE[["grey"]]) +
  scale_x_log10(sec.axis = sec_axis(~., breaks = c(1e-5, 1e-3, 1e-1), labels = sprintf("%.0f", sapply(c(1e-5, 1e-3, 1e-1), function(t) loo(t)[["df"]])),
                                    name = expression("degrees of freedom tr" ~ H[lambda]))) + scale_y_log10() +
  labs(x = expression("penalty" ~ lambda), y = "leave-one-out MSE", title = "Leave-one-out error",
       caption = "line: closed form; circles: refitting n times") + theme_blog()
save_fig(p3 + p4, here, "penalty-selection", width = 7.8, height = 3.6)
