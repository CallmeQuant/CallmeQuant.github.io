# Consistency of maximum likelihood: Cauchy location criterion and MLE paths (figure 1);
# unbounded likelihood of a normal mixture with a free scale (figure 2).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(1949)
avg_llr <- function(x, th) sapply(th, function(t) -mean(log1p((x - t)^2) - log1p(x^2)))
cauchy_mle <- function(x) {
  grid <- seq(-15, 15, length.out = 3001)
  g <- grid[which.max(avg_llr(x, grid))]
  optimize(function(t) avg_llr(x, t), c(g - 0.02, g + 0.02), maximum = TRUE)$maximum
}

# ---- Figure 1 ----
x <- rcauchy(5000)
th <- seq(-6, 6, length.out = 601)
crit <- rbind(
  do.call(rbind, lapply(c(10, 100, 1000), function(n)
    data.frame(theta = th, value = avg_llr(x[1:n], th), curve = paste0("n = ", n)))),
  data.frame(theta = th, value = -log1p(th^2 / 4), curve = "limit")
)
crit$curve <- factor(crit$curve, levels = c("n = 10", "n = 100", "n = 1000", "limit"))
p1 <- ggplot(crit, aes(theta, value, colour = curve, linetype = curve)) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = unname(c(PALETTE[c("gold", "green", "blue", "ink")])),
                      labels = c(expression(M[n](theta) ~ "," ~ n == 10), expression(M[n](theta) ~ "," ~ n == 100),
                                 expression(M[n](theta) ~ "," ~ n == 1000), expression(M(theta) == -log(1 + theta^2 / 4)))) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed"), guide = "none") +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-2.6, 1.3)) +
  labs(x = expression(theta), y = "average log-likelihood ratio",
       title = expression("Criterion and its limit (" * theta[0] == 0 * ")")) +
  theme_blog() + theme(legend.position = c(0.25, 0.86), legend.text = element_text(size = 7.5),
                       legend.key.height = unit(0.8, "lines"))

ns <- unique(round(10^seq(0.7, log10(5000), length.out = 70)))
paths <- do.call(rbind, lapply(1:5, function(j) {
  xs <- if (j == 1) x else rcauchy(5000)
  data.frame(n = ns, est = sapply(ns, function(n) cauchy_mle(xs[1:n])), path = factor(j))
}))
p2 <- ggplot(paths, aes(n, est, colour = path)) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.35) +
  geom_line(linewidth = 0.5) +
  scale_colour_manual(values = CYCLE[1:5], guide = "none") +
  scale_x_log10() + coord_cartesian(ylim = c(-3, 3)) +
  labs(x = "sample size n (log scale)", y = expression(hat(theta)[n]), title = "Five paths of the global maximizer") +
  theme_blog()
save_fig(p1 + p2 + plot_layout(widths = c(1.15, 1)), here, "cauchy-criterion", width = 7.8, height = 3.5)

# ---- Figure 2 ----
n <- 200
z <- runif(n) < 0.5
y <- ifelse(z, rnorm(n, 2, 1), rnorm(n, 0, 1))
loglik <- function(mu, logsig) {
  cc <- -0.5 * log(2 * pi) + log(0.5)
  a <- cc - y^2 / 2
  q <- exp(pmin(2 * log(abs(y - mu)) - 2 * logsig, 700))
  b <- cc - logsig - 0.5 * q
  m <- pmax(a, b)
  sum(m + log(exp(a - m) + exp(b - m)))
}
mk <- function(lo, npts, panel) {
  ls <- seq(lo, 1.5, length.out = npts)
  rbind(data.frame(ls = ls, ll = sapply(ls, function(t) loglik(2, t)), mu = "true", panel = panel),
        data.frame(ls = ls, ll = sapply(ls, function(t) loglik(y[1], t)), mu = "Y1", panel = panel))
}
d2 <- rbind(mk(-8, 600, "Near~the~true~scale"), mk(-500, 2000, "Down~to~sigma == e^{-500}"))
d2$panel <- factor(d2$panel, levels = c("Near~the~true~scale", "Down~to~sigma == e^{-500}"))
d2$mu <- factor(d2$mu, levels = c("true", "Y1"))
lmax <- -optimize(function(t) -loglik(2, t), c(-2, 1.5))$objective   # local maximum of the mu = 2 curve
p3 <- ggplot(d2, aes(ls, ll, colour = mu)) +
  geom_hline(yintercept = lmax, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~panel, scales = "free_x", labeller = label_parsed) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]),
                      labels = c(expression(mu == 2 ~ "(true value)"), expression(mu == Y[1]))) +
  labs(x = expression(log ~ sigma), y = expression("log-likelihood" ~ l[n](mu, sigma)),
       title = expression("Model" ~ frac(1, 2) * N(0, 1) + frac(1, 2) * N(mu, sigma^2) * ", data with" ~ mu == 2 * "," ~ sigma == 1 * "," ~ n == 200)) +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p3, here, "mixture-degeneracy", width = 7.8, height = 3.6)
