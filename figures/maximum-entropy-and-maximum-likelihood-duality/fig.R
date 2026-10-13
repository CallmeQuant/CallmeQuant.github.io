# Duality between maximum entropy and maximum likelihood on X = {0,1,2,3}, T(x) = x.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(7)
x <- 0:3
true_p <- c(0.15, 0.45, 0.10, 0.30)
n <- 30
dat <- sample(x, n, replace = TRUE, prob = true_p)
p_hat <- tabulate(dat + 1, nbins = 4) / n
t_hat <- mean(dat)
A <- function(th) sapply(th, function(t) log(sum(exp(t * x))))
mean_of <- function(th) sum(x * exp(th * x)) / sum(exp(th * x))
pmf <- function(th) exp(th * x - A(th))
H <- function(p) -sum(p[p > 0] * log(p[p > 0]))
theta_hat <- uniroot(function(t) mean_of(t) - t_hat, c(-30, 30), tol = 1e-12)$root
g_min <- A(theta_hat) - theta_hat * t_hat

th <- seq(-2.5, 2, length.out = 400)
pa <- ggplot(data.frame(th = th, g = A(th) - th * t_hat), aes(th, g)) +
  geom_line(colour = PALETTE[["blue"]], linewidth = 0.8) +
  geom_hline(yintercept = H(p_hat), colour = PALETTE[["red"]], linetype = "dashed") +
  geom_hline(yintercept = g_min, colour = PALETTE[["ink"]], linetype = "dotted", linewidth = 0.3) +
  annotate("point", x = theta_hat, y = g_min, colour = PALETTE[["ink"]], size = 2) +
  annotate("text", x = theta_hat + 0.1, y = g_min - 0.08, label = "hat(theta)", parse = TRUE, size = 3.4) +
  annotate("text", x = -2.4, y = H(p_hat) - 0.08, label = "H(hat(p)[n])", parse = TRUE,
           colour = PALETTE[["red"]], hjust = 0, size = 3.4) +
  coord_cartesian(xlim = c(-2.5, 2), ylim = c(0.9, 2.6)) +
  labs(x = expression(theta), y = NULL, title = expression("Dual: " * A(theta) - theta * hat(t) == -l[n](theta))) +
  theme_blog()

ms <- seq(0.02, 2.98, length.out = 300)
Hstar <- sapply(ms, function(m) H(pmf(uniroot(function(t) mean_of(t) - m, c(-40, 40), tol = 1e-12)$root)))
lines_df <- do.call(rbind, lapply(c(-1.5, -0.6, 0.7, 1.6), function(t0) data.frame(m = ms, y = A(t0) - t0 * ms, t0 = t0)))
pb <- ggplot() +
  geom_line(data = lines_df, aes(m, y, group = t0), colour = PALETTE[["grey"]], linewidth = 0.4) +
  geom_line(aes(ms, A(theta_hat) - theta_hat * ms), colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.4) +
  geom_line(aes(ms, Hstar), colour = PALETTE[["blue"]], linewidth = 0.9) +
  annotate("point", x = t_hat, y = H(pmf(theta_hat)), colour = PALETTE[["ink"]], size = 2) +
  annotate("point", x = t_hat, y = H(p_hat), colour = PALETTE[["red"]], size = 2) +
  annotate("text", x = t_hat + 0.08, y = H(p_hat) - 0.02, label = "hat(p)[n]", parse = TRUE,
           colour = PALETTE[["red"]], hjust = 0, size = 3.4) +
  annotate("text", x = t_hat - 0.1, y = H(pmf(theta_hat)) + 0.12, label = "p[hat(theta)]", parse = TRUE, size = 3.4) +
  coord_cartesian(xlim = c(0, 3), ylim = c(0, 1.75)) +
  labs(x = "mean m", y = "entropy (nats)", title = expression("Primal: lines " * m %->% A(theta) - theta * m)) +
  theme_blog()

dc <- rbind(data.frame(x = x, p = p_hat, w = "empirical"), data.frame(x = x, p = pmf(theta_hat), w = "fitted"))
pc <- ggplot(dc, aes(x, p, fill = w)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.76) +
  scale_fill_manual(values = unname(PALETTE[c("red", "blue")]),
                    labels = c(expression(hat(p)[n]), expression(p[hat(theta)]))) +
  coord_cartesian(ylim = c(0, 0.62)) +
  labs(x = "x", y = "probability", title = bquote("Same mean" ~ hat(t) == .(sprintf("%.2f", t_hat)))) +
  theme_blog() + theme(legend.position = c(0.25, 0.92), legend.direction = "horizontal",
                       panel.grid.major.x = element_blank())
save_fig(pa + pb + pc + plot_layout(widths = c(1.05, 1.15, 0.9)), here, "duality", width = 9.8, height = 3.3)
