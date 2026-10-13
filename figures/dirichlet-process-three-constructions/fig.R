# Dirichlet process DP(alpha, N(0,1)): random distribution functions for three concentration
# parameters (stick-breaking), stick-breaking weights, and the number of distinct values in a
# Polya urn sequence.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

stick_breaking <- function(alpha, tol = 1e-10) {
  w <- c(); atoms <- c(); rest <- 1
  while (rest > tol) {
    v <- rbeta(1, 1, alpha); w <- c(w, rest * v); atoms <- c(atoms, rnorm(1)); rest <- rest * (1 - v)
  }
  list(w = w, atoms = atoms)
}

set.seed(1973)
d1 <- do.call(rbind, lapply(c(1, 10, 100), function(alpha) do.call(rbind, lapply(1:8, function(j) {
  s <- stick_breaking(alpha); o <- order(s$atoms)
  data.frame(x = c(-3, s$atoms[o], 3), F = c(0, cumsum(s$w[o]), sum(s$w)), draw = j,
             alpha = paste("alpha ==", alpha))
}))))
d1$alpha <- factor(d1$alpha, levels = paste("alpha ==", c(1, 10, 100)))
base <- data.frame(x = seq(-3, 3, length.out = 400)); base$F <- pnorm(base$x)
p1 <- ggplot(d1, aes(x, F, group = draw)) +
  geom_step(colour = PALETTE[["blue"]], alpha = 0.7, linewidth = 0.35) +
  geom_line(data = base, aes(x, F), inherit.aes = FALSE, colour = PALETTE[["red"]], linetype = "dashed", linewidth = 0.7) +
  facet_wrap(~alpha, labeller = label_parsed) +
  coord_cartesian(xlim = c(-3, 3), ylim = c(0, 1), expand = FALSE) +
  labs(x = "x", y = "P((\u2212\u221e, x])") + theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "dp-cdfs", width = 8, height = 2.9)

set.seed(1994)
K <- 25
d2 <- do.call(rbind, lapply(c(2, 20), function(alpha) {
  s <- stick_breaking(alpha)
  data.frame(k = 1:K, w = s$w[1:K], Ew = (1 / (1 + alpha)) * (alpha / (1 + alpha))^(0:(K - 1)),
             alpha = paste("alpha ==", alpha))
}))
d2$alpha <- factor(d2$alpha, levels = paste("alpha ==", c(2, 20)))
p2 <- ggplot(d2, aes(k, w, fill = alpha)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.76) +
  geom_line(aes(k, Ew, colour = alpha), linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = CYCLE[1:2], labels = scales::label_parse()) +
  scale_colour_manual(values = CYCLE[1:2], guide = "none") +
  coord_cartesian(xlim = c(0.3, K + 0.7), ylim = c(0, 0.6), expand = FALSE) +
  labs(x = "index k", y = expression("weight" ~ w[k]), title = "Stick-breaking weights") +
  theme_blog() + theme(legend.position = c(0.8, 0.85), legend.justification = c(0.5, 0.5),
                       panel.grid.major.x = element_blank())

set.seed(1953)
N <- 5000; n <- 1:N
paths <- do.call(rbind, lapply(c(1, 5, 25), function(alpha) do.call(rbind, lapply(1:5, function(r)
  data.frame(n = n, K = cumsum(runif(N) < alpha / (alpha + n - 1)), path = paste(alpha, r),
             alpha = paste("alpha ==", alpha))))))
expct <- do.call(rbind, lapply(c(1, 5, 25), function(alpha)
  data.frame(n = n, K = cumsum(alpha / (alpha + n - 1)), alpha = paste("alpha ==", alpha))))
lv <- paste("alpha ==", c(1, 5, 25)); paths$alpha <- factor(paths$alpha, lv); expct$alpha <- factor(expct$alpha, lv)
p3 <- ggplot() +
  geom_line(data = paths, aes(n, K, group = path, colour = alpha), alpha = 0.5, linewidth = 0.3) +
  geom_line(data = expct, aes(n, K, colour = alpha), linewidth = 0.9) +
  scale_colour_manual(values = CYCLE[1:3], labels = scales::label_parse()) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^(0:3)) +
  coord_cartesian(xlim = c(1, N), ylim = c(0, 180), expand = FALSE) +
  labs(x = "sample size n (log scale)", y = expression("distinct values" ~ K[n]), title = "Number of distinct values") +
  theme_blog() + theme(legend.position = c(0.2, 0.8), legend.justification = c(0.5, 0.5))
save_fig(p2 + p3, here, "dp-weights-clusters", width = 7.6, height = 3.2)
