# Posterior consistency for a Bernoulli probability: contraction under a uniform prior, exponential
# decay of the posterior mass outside a neighbourhood, and concentration at the Kullback-Leibler
# projection when the prior support excludes the truth.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

theta0 <- 0.3; eps <- 0.05

set.seed(3)
S <- cumsum(runif(10000) < theta0)
t <- seq(0, 1, length.out = 2001)
d1 <- do.call(rbind, lapply(c(10, 100, 1000, 10000), function(n)
  data.frame(t = t, dens = dbeta(t, 1 + S[n], 1 + n - S[n]), n = paste("n =", n))))
d1$n <- factor(d1$n, levels = paste("n =", c(10, 100, 1000, 10000)))
p1 <- ggplot(d1, aes(t, dens, colour = n)) +
  geom_vline(xintercept = theta0, colour = PALETTE[["grey"]], linetype = "dotted") +
  geom_line(linewidth = 0.7) + scale_colour_manual(values = CYCLE[1:4]) +
  coord_cartesian(xlim = c(0, 0.7), ylim = c(0, 95), expand = FALSE) +
  labs(x = expression(theta), y = "posterior density", title = "Posterior along one sample path") +
  theme_blog() + theme(legend.position = c(0.8, 0.75), legend.justification = c(0.5, 0.5))

ns <- 1:3000
d2 <- do.call(rbind, lapply(1:8, function(j) {
  Sj <- cumsum(runif(length(ns)) < theta0)
  a <- 1 + Sj; b <- 1 + ns - Sj
  data.frame(n = ns, out = pmax(pbeta(theta0 - eps, a, b) + pbeta(theta0 + eps, a, b, lower.tail = FALSE), 1e-300), path = j)
}))
p2 <- ggplot(d2, aes(n, out, group = path)) +
  geom_line(colour = PALETTE[["blue"]], linewidth = 0.35, alpha = 0.75) +
  scale_y_log10(breaks = 10^seq(-10, 0, 2), labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(xlim = c(0, 3000), ylim = c(1e-10, 3), expand = FALSE) +
  labs(x = "sample size n", y = expression(Pi[n](paste("|", theta - 0.3, "|") > 0.05)),
       title = "Mass outside a neighbourhood") +
  theme_blog()
save_fig(p1 + p2, here, "contraction", width = 7.6, height = 3.3)

kl <- function(t0, t) t0 * log(t0 / t) + (1 - t0) * log((1 - t0) / (1 - t))
tt <- seq(0.001, 0.999, length.out = 2000)
in_supp <- tt <= 0.2 | tt >= 0.5
set.seed(99)
S <- cumsum(runif(400) < theta0)
p3 <- ggplot(data.frame(t = tt, k = kl(theta0, tt)), aes(t, k)) +
  annotate("rect", xmin = 0, xmax = 0.2, ymin = -Inf, ymax = Inf, fill = PALETTE[["light"]]) +
  annotate("rect", xmin = 0.5, xmax = 1, ymin = -Inf, ymax = Inf, fill = PALETTE[["light"]]) +
  geom_vline(xintercept = theta0, colour = PALETTE[["grey"]], linetype = "dotted") +
  geom_line(colour = PALETTE[["ink"]], linewidth = 0.7) +
  annotate("point", x = c(0.2, 0.5), y = kl(theta0, c(0.2, 0.5)), colour = PALETTE[c("red", "grey")], size = 2.4) +
  annotate("segment", x = 0.2, y = kl(theta0, 0.2), xend = 0.33, yend = 0.17, linewidth = 0.25) +
  annotate("text", x = 0.33, y = 0.19, label = "theta^'*' == 0.2", parse = TRUE, hjust = 0, size = 3.4) +
  annotate("text", x = 0.31, y = 0.64, label = "theta[0]", parse = TRUE, hjust = 0, size = 3.4) +
  annotate("text", x = 0.64, y = 0.64, label = "prior support", size = 3.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 0.7), expand = FALSE) +
  labs(x = expression(theta), y = expression(K(theta[0], theta)), title = "Divergence from the truth") +
  theme_blog() + theme(panel.grid.major = element_blank())

d4 <- do.call(rbind, lapply(c(25, 100, 400), function(n) {
  s <- S[n]; lp <- s * log(tt) + (n - s) * log1p(-tt); lp[!in_supp] <- -Inf
  w <- exp(lp - max(lp[in_supp])); w <- w / sum(diff(tt) * (head(w, -1) + tail(w, -1)) / 2)
  data.frame(t = tt, dens = w, n = paste("n =", n))
}))
d4$n <- factor(d4$n, levels = paste("n =", c(25, 100, 400)))
p4 <- ggplot(d4, aes(t, dens, colour = n)) +
  geom_vline(xintercept = theta0, colour = PALETTE[["grey"]], linetype = "dotted") +
  geom_line(linewidth = 0.7) + scale_colour_manual(values = CYCLE[1:3]) +
  coord_cartesian(xlim = c(0, 0.7), ylim = c(0, 165), expand = FALSE) +
  labs(x = expression(theta), y = "posterior density", title = expression("Posterior concentrates at" ~ theta^"*")) +
  theme_blog() + theme(legend.position = c(0.8, 0.8), legend.justification = c(0.5, 0.5))
save_fig(p3 + p4, here, "misspecified-support", width = 7.6, height = 3.3)
