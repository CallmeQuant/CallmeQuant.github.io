# Score matching for the unnormalized double-well family log q(x) = -t1 x^2/2 - t2 x^4/4,
# compared with maximum likelihood.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

theta0 <- c(-2, 1)
logq <- function(x, t) -t[1] * x^2 / 2 - t[2] * x^4 / 4
grid <- seq(-5, 5, length.out = 20001); dx <- grid[2] - grid[1]

sample_q <- function(n) {                       # exact draws by rejection from N(0, 1.5^2)
  s <- 1.5; logM <- max(logq(grid, theta0) + grid^2 / (2 * s^2)); out <- numeric(0)
  while (length(out) < n) {
    x <- rnorm(2 * n, 0, s)
    out <- c(out, x[log(runif(2 * n)) < logq(x, theta0) + x^2 / (2 * s^2) - logM])
  }
  out[1:n]
}
score_matching <- function(x) {
  A <- matrix(c(mean(x^2), mean(x^4), mean(x^4), mean(x^6)), 2)
  -solve(A, c(-1, -3 * mean(x^2)))
}
log_Z <- function(t) { v <- logq(grid, t); m <- max(v); m + log(sum(exp(v - m)) * dx) }
mle <- function(x, start) {
  m2 <- mean(x^2); m4 <- mean(x^4)
  nll <- function(t) if (t[2] > 0) t[1] * m2 / 2 + t[2] * m4 / 4 + log_Z(t) else Inf
  optim(start, nll, control = list(reltol = 1e-12, maxit = 2000))$par
}

set.seed(2005)
x <- sample_q(1000)
t_sm <- score_matching(x); t_ml <- mle(x, t_sm)
xs <- seq(-3.2, 3.2, length.out = 600)
labs3 <- c(sprintf("true (%.0f, %.0f)", theta0[1], theta0[2]),
           sprintf("score matching (%.2f, %.2f)", t_sm[1], t_sm[2]),
           sprintf("maximum likelihood (%.2f, %.2f)", t_ml[1], t_ml[2]))
dens <- do.call(rbind, Map(function(t, lab) data.frame(x = xs, d = exp(logq(xs, t) - log_Z(t)), fit = lab),
                           list(theta0, t_sm, t_ml), labs3))
dens$fit <- factor(dens$fit, labs3)
pa <- ggplot() +
  geom_histogram(data = data.frame(x = x), aes(x, after_stat(density)), breaks = seq(-3.2, 3.2, length.out = 50),
                 fill = PALETTE[["light"]], colour = "white", linewidth = 0.2) +
  geom_line(data = dens, aes(x, d, colour = fit, linetype = fit), linewidth = 0.6) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "blue", "red")])) +
  scale_linetype_manual(values = c(1, 1, 2)) +
  coord_cartesian(xlim = c(-3.2, 3.2), ylim = c(0, 0.62)) +
  labs(x = "x", y = "density", title = expression(n == 1000 ~ "draws from" ~ q(x) %prop% e^{x^2 - x^4/4})) +
  theme_blog() + theme(legend.position = c(0.5, 0.99), legend.justification = c(0.5, 1), legend.text = element_text(size = 8))

R <- 400; n <- 500
est <- do.call(rbind, lapply(seq_len(R), function(r) {
  x <- sample_q(n); s <- score_matching(x); m <- mle(x, s)
  rbind(data.frame(t1 = s[1], t2 = s[2], method = "score matching"),
        data.frame(t1 = m[1], t2 = m[2], method = "maximum likelihood"))
}))
est$method <- factor(est$method, c("score matching", "maximum likelihood"))
pb <- ggplot(est, aes(t1, t2, colour = method)) +
  geom_point(size = 0.8, alpha = 0.5) +
  stat_ellipse(type = "norm", level = 0.95, linewidth = 0.6) +
  annotate("point", x = theta0[1], y = theta0[2], shape = 3, size = 4, stroke = 1.1, colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  labs(x = expression(hat(theta)[1]), y = expression(hat(theta)[2]), title = sprintf("%d data sets of size n = %d", R, n)) +
  theme_blog() + theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0), legend.text = element_text(size = 8))
save_fig(pa + pb + plot_layout(widths = c(1.1, 1)), here, "double-well", width = 7.8, height = 3.5)
