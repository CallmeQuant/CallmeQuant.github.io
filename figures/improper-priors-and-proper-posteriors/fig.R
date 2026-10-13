# Improper priors: collapse of Beta(eps, eps) posteriors when X = 0, and the posterior of log(tau)
# in a normal hierarchical model under the priors 1/tau and 1.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

n <- 10
t <- seq(0, 0.5, length.out = 600)
eps_values <- c(1, 0.3, 0.1, 0.03, 0.01)
d1 <- do.call(rbind, lapply(eps_values, function(e)
  data.frame(t = t, cdf = pbeta(t, e, n + e), eps = paste0("epsilon == ", e))))
d1$eps <- factor(d1$eps, levels = paste0("epsilon == ", eps_values))

y <- c(2.1, -0.4, 1.3, 0.2, 3.0, -1.1, 0.9, 1.7)
s <- c(1.0, 0.8, 1.5, 1.2, 1.0, 2.0, 0.9, 1.1)
log_marg <- function(tau) sapply(tau, function(tt) {
  w <- 1 / (s^2 + tt^2); W <- sum(w); mu <- sum(w * y) / W
  -0.5 * (length(y) - 1) * log(2 * pi) - 0.5 * log(W) + 0.5 * sum(log(w)) - 0.5 * sum(w * (y - mu)^2)
})
logtau <- seq(log(1e-4), log(1e2), length.out = 800)
lm <- log_marg(exp(logtau))
lu <- lm + logtau
d2 <- rbind(
  data.frame(x = logtau / log(10), dens = exp(lm - max(lm)), prior = "pi(tau) %prop% 1/tau"),
  data.frame(x = logtau / log(10), dens = exp(lu - max(lu)), prior = "pi(tau) %prop% 1")
)
d2$prior <- factor(d2$prior, levels = c("pi(tau) %prop% 1/tau", "pi(tau) %prop% 1"))

p1 <- ggplot(d1, aes(t, cdf, colour = eps)) + geom_line(linewidth = 0.8) +
  scale_colour_manual(values = CYCLE[1:5], labels = scales::label_parse()) +
  coord_cartesian(xlim = c(0, 0.5), ylim = c(0, 1.03), expand = FALSE) +
  labs(x = "t", y = expression(P(paste(theta <= t, " | ", X == 0))),
       title = expression(paste("Beta(", epsilon, ", ", epsilon, ") prior, ", n == 10, ", ", X == 0))) +
  theme_blog() + theme(legend.position = c(0.78, 0.3), legend.justification = c(0.5, 0.5))
p2 <- ggplot(d2, aes(x, dens, colour = prior)) + geom_line(linewidth = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")]), labels = scales::label_parse()) +
  scale_x_continuous(breaks = -4:2, labels = scales::label_math(10^.x)) +
  coord_cartesian(xlim = c(-4, 2), ylim = c(0, 1.08), expand = FALSE) +
  labs(x = expression(tau ~ "(log scale)"), y = expression("density of" ~ log(tau) ~ "(unnormalized)"),
       title = "Hierarchical model, J = 8 groups") +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
save_fig(p1 + p2, here, "improper-limits", width = 7.6, height = 3.3)
