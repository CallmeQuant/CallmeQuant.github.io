# EM for the symmetric mixture 0.5 N(theta, 1) + 0.5 N(-theta, 1): lower bounds touching the
# log-likelihood, and linear convergence at the missing-information rate.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(1977); n <- 200
simulate <- function(ts) sample(c(-1, 1), n, replace = TRUE) * ts + rnorm(n)
lse <- function(a, b) pmax(a, b) + log1p(exp(-abs(a - b)))
loglik <- function(th, x) sapply(th, function(t) sum(lse(-0.5 * (x - t)^2, -0.5 * (x + t)^2) - log(2) - 0.5 * log(2 * pi)))
lower_bound <- function(th, x, tt) {
  r <- 1 / (1 + exp(-2 * tt * x)); ent <- -(r * log(r) + (1 - r) * log1p(-r))
  sapply(th, function(t) sum(r * (-0.5 * (x - t)^2) + (1 - r) * (-0.5 * (x + t)^2) - log(2) - 0.5 * log(2 * pi) + ent))
}
em_map <- function(t, x) mean(x * tanh(t * x))

x <- simulate(1.0)
it <- 0.15; for (k in 1:4) it <- c(it, em_map(it[k], x))
grid <- seq(-0.2, 1.8, length.out = 400); ll <- loglik(grid, x)
d1 <- rbind(data.frame(theta = grid, v = ll, what = "italic(l)(theta)"),
            do.call(rbind, lapply(1:4, function(t) data.frame(theta = grid, v = lower_bound(grid, x, it[t]),
                                                               what = sprintf("F(q[%d], theta)", t)))))
d1$what <- factor(d1$what, levels = unique(d1$what))
pts <- data.frame(theta = it[1:4], v = loglik(it[1:4], x), what = levels(d1$what)[2:5])
tri <- data.frame(theta = it[2:5], v = mapply(function(a, b) lower_bound(a, x, b), it[2:5], it[1:4]), what = levels(d1$what)[2:5])
p1 <- ggplot(d1, aes(theta, v, colour = what, linetype = what)) + geom_line(linewidth = 0.7) +
  geom_point(data = pts, size = 1.8, show.legend = FALSE) + geom_point(data = tri, shape = 17, size = 2, show.legend = FALSE) +
  scale_colour_manual(values = c(PALETTE[["ink"]], CYCLE[1:4]), labels = scales::label_parse()) +
  scale_linetype_manual(values = c("solid", rep("dashed", 4)), labels = scales::label_parse()) +
  coord_cartesian(xlim = c(-0.2, 1.8), ylim = max(ll) + c(-70, 8), expand = FALSE) +
  labs(x = expression(theta), y = "log-likelihood and lower bounds", title = expression("Bounds touching" ~ italic(l) ~ "at" ~ theta[t])) +
  guides(colour = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)) + theme_blog()

d2 <- do.call(rbind, lapply(c(0.6, 1.0, 2.0), function(ts) {
  xs <- simulate(ts); th <- if (ts > 0.7) 0.5 * ts else 1.2
  h <- th; for (k in 1:400) h <- c(h, em_map(h[k], xs))
  hat <- h[401]; rate <- mean(xs^2 / cosh(hat * xs)^2)
  err <- abs(h[1:60] - hat)
  data.frame(t = 0:59, err = err, pred = err[4] * rate^((0:59) - 3),
             lab = sprintf("theta^'*' == '%.1f' * ':' ~ 'rate' ~ hat(rho) == '%.2f'", ts, rate))
}))
p2 <- ggplot(subset(d2, err > 1e-13), aes(t, err, colour = lab)) + geom_point(size = 1.1) +
  geom_line(data = d2, aes(y = pred), linetype = "dotted") +
  scale_y_log10(limits = c(1e-12, 3)) + scale_colour_blog(labels = scales::label_parse()) +
  labs(x = expression("iteration" ~ t), y = expression("|" * theta[t] - hat(theta) * "|"), title = "Linear convergence") +
  guides(colour = guide_legend(ncol = 1)) + theme_blog()
save_fig(p1 + p2, here, "em-bounds-and-rates", width = 8.4, height = 4.0)
