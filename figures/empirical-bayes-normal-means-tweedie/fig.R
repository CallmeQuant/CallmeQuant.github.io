# Empirical Bayes for p = 1000 normal means with a two-component prior: the marginal density of X
# and its log-polynomial estimate, and the estimated rules (James-Stein and Tweedie) against the
# oracle posterior mean.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

p <- 1000; w1 <- 0.3; mu1 <- 3; s1 <- 0.5
oracle <- function(x) {
  f0 <- (1 - w1) * dnorm(x); v <- 1 + s1^2
  f1 <- w1 * dnorm(x, mu1, sqrt(v))
  f1 * (mu1 + s1^2 / v * (x - mu1)) / (f0 + f1)
}
marginal <- function(x) (1 - w1) * dnorm(x) + w1 * dnorm(x, mu1, sqrt(1 + s1^2))
tweedie <- function(x, at, degree = 5, nbins = 60) {
  edges <- seq(min(x) - 1e-9, max(x) + 1e-9, length.out = nbins + 1)
  mids <- (head(edges, -1) + tail(edges, -1)) / 2; width <- edges[2] - edges[1]
  y <- hist(x, breaks = edges, plot = FALSE)$counts
  cc <- mean(mids); ss <- sqrt(mean((mids - cc)^2))
  Z <- outer((mids - cc) / ss, 0:degree, `^`)
  fit <- glm.fit(Z, y, family = poisson())
  b <- fit$coefficients
  u <- (at - cc) / ss
  logf <- drop(outer(u, 0:degree, `^`) %*% b)
  dlogf <- drop(outer(u, 0:(degree - 1), `^`) %*% (b[-1] * (1:degree))) / ss
  list(est = at + dlogf, f = exp(logf) / (length(x) * width))
}

set.seed(1956)
z <- runif(p) < w1
theta <- ifelse(z, rnorm(p, mu1, s1), 0)
x <- theta + rnorm(p)
grid <- seq(-3.5, 6.5, length.out = 600)
tw <- tweedie(x, grid)
xb <- mean(x); b <- min(1, (p - 3) / sum((x - xb)^2))

d1 <- rbind(data.frame(x = grid, y = marginal(grid), what = "true f"),
            data.frame(x = grid, y = tw$f, what = "fitted f (Poisson regression)"))
d1$what <- factor(d1$what, levels = c("true f", "fitted f (Poisson regression)"))
p1 <- ggplot() +
  geom_histogram(data = data.frame(x = x), aes(x, after_stat(density)), bins = 40,
                 fill = PALETTE[["light"]], colour = "white", linewidth = 0.2) +
  geom_line(data = d1, aes(x, y, colour = what, linetype = what), linewidth = 0.7) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red")]), labels = c(expression("true" ~ italic(f)), expression("Poisson-regression fit" ~ hat(italic(f))))) +
  scale_linetype_manual(values = c("solid", "dashed"), labels = c(expression("true" ~ italic(f)), expression("Poisson-regression fit" ~ hat(italic(f))))) +
  coord_cartesian(xlim = c(-3.5, 6.5), ylim = c(0, 0.36), expand = FALSE) +
  labs(x = expression(italic(x)), y = "density", title = expression("Marginal density of" ~ italic(X) * "," ~ italic(p) == 1000)) +
  theme_blog() + theme(legend.position = c(0.7, 0.88), legend.justification = c(0.5, 0.5))

inside <- grid >= quantile(x, 0.005) & grid <= quantile(x, 0.995)
lv <- c("MLE", "James-Stein (linear)", "Tweedie, estimated", "oracle Bayes rule")
d2 <- rbind(data.frame(x = grid, y = grid, what = lv[1]),
            data.frame(x = grid, y = xb + (1 - b) * (grid - xb), what = lv[2]),
            data.frame(x = grid[inside], y = tw$est[inside], what = lv[3]),
            data.frame(x = grid, y = oracle(grid), what = lv[4]))
d2$what <- factor(d2$what, levels = lv)
p2 <- ggplot(d2, aes(x, y, colour = what, linetype = what)) + geom_line(linewidth = 0.7) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "green", "red", "ink")])) +
  scale_linetype_manual(values = c("dotted", "solid", "dashed", "solid")) +
  coord_cartesian(xlim = c(-3.5, 6.5), ylim = c(-1.5, 5.5), expand = FALSE) +
  labs(x = expression(italic(x)), y = expression("estimate of" ~ theta), title = "Estimated posterior means") +
  theme_blog() + theme(legend.position = c(0.32, 0.78), legend.justification = c(0.5, 0.5))
save_fig(p1 + p2, here, "eb-rules", width = 7.6, height = 3.4)
