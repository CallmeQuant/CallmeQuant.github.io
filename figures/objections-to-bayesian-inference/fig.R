# Stopping rules and the likelihood principle: three successes in twelve Bernoulli trials under a
# fixed-n (binomial) design and a sample-until-third-success (negative binomial) design, and the
# common posterior under a uniform prior.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

d1 <- data.frame(k = 0:12, p = dbinom(0:12, 12, 0.5)); d1$tail <- d1$k <= 3
d2 <- data.frame(n = 3:25); d2$p <- choose(d2$n - 1, 2) * 0.5^d2$n; d2$tail <- d2$n >= 12
fills <- c(`TRUE` = PALETTE[["red"]], `FALSE` = PALETTE[["light"]])

p1 <- ggplot(d1, aes(k, p, fill = tail)) + geom_col(width = 0.75) +
  scale_fill_manual(values = fills, guide = "none") +
  scale_x_continuous(breaks = c(0, 3, 6, 9, 12)) +
  coord_cartesian(ylim = c(0, 0.25), expand = FALSE) +
  labs(x = "successes X in n = 12", y = expression("probability under" ~ theta == 1/2),
       title = expression("Binomial:" ~ P(X <= 3) == 0.073)) +
  theme_blog() + theme(panel.grid.major.x = element_blank())
p2 <- ggplot(d2, aes(n, p, fill = tail)) + geom_col(width = 0.75) +
  scale_fill_manual(values = fills, guide = "none") +
  scale_x_continuous(breaks = seq(3, 24, 3)) +
  coord_cartesian(ylim = c(0, 0.25), expand = FALSE) +
  labs(x = "trial N of the third success", y = NULL,
       title = expression("Neg. binomial:" ~ P(N >= 12) == 0.033)) +
  theme_blog() + theme(panel.grid.major.x = element_blank())
t <- seq(0, 1, length.out = 600)
d3 <- data.frame(t = t, d = dbeta(t, 4, 10))
p3 <- ggplot(d3, aes(t, d)) +
  geom_area(data = subset(d3, t >= 0.5), fill = PALETTE[["red"]]) +
  geom_line(colour = PALETTE[["blue"]], linewidth = 0.8) +
  annotate("segment", x = 0.53, y = 0.25, xend = 0.55, yend = 2.65, linewidth = 0.25) +
  annotate("text", x = 0.45, y = 2.8, label = "P(theta >= 1/2) == 0.046", parse = TRUE, hjust = 0, size = 3.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 3.6), expand = FALSE) +
  labs(x = expression(theta), y = "posterior density", title = "Posterior: Beta(4, 10)") +
  theme_blog()
save_fig(p1 + p2 + p3 & theme(plot.title = element_text(size = 9.5)), here, "stopping-rule", width = 8.4, height = 3.0)
