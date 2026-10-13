# Lindley's paradox: posterior probability of a point null at a fixed z-statistic, and the
# Bayesian critical value of |z| as functions of n (tau = sigma, pi0 = 1/2).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

n <- 10^seq(0, 7, length.out = 600)
r <- n
zs <- c(1.96, 2.576, 3.291)
labs_z <- c("z = 1.96 (p = 0.05)", "z = 2.58 (p = 0.01)", "z = 3.29 (p = 0.001)")
d1 <- do.call(rbind, lapply(1:3, function(k) {
  B <- sqrt(1 + r) * exp(-0.5 * zs[k]^2 * r / (1 + r))
  data.frame(n = n, post = B / (1 + B), z = labs_z[k])
}))
d1$z <- factor(d1$z, levels = labs_z)
lb <- data.frame(z = factor(labs_z, levels = labs_z), lb = 1 / (1 + exp(zs^2 / 2)))

p1 <- ggplot(d1, aes(n, post, colour = z)) +
  geom_hline(data = lb, aes(yintercept = lb, colour = z), linetype = "dashed", linewidth = 0.35) +
  geom_line(linewidth = 0.8) +
  scale_x_log10(breaks = 10^(0:7), labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = CYCLE[1:3]) +
  coord_cartesian(xlim = c(1, 1e7), ylim = c(0, 1), expand = FALSE) +
  labs(x = "sample size n", y = expression(P(H[0] ~ "|" ~ X)), title = "Posterior probability of H0 at fixed z") +
  theme_blog() + theme(legend.position = "bottom", legend.direction = "vertical")

d2 <- data.frame(n = n, c = sqrt((1 + 1 / r) * log1p(r)))
ref <- data.frame(y = c(1.96, 3.291), lab = c("5% test", "0.1% test"))
ref$lab <- factor(ref$lab, levels = c("Bayes: P(H0 | X) = 1/2", "5% test", "0.1% test"))
p2 <- ggplot() +
  geom_line(data = d2, aes(n, c, colour = "Bayes: P(H0 | X) = 1/2"), linewidth = 0.8) +
  geom_hline(data = ref, aes(yintercept = y, colour = lab), linetype = "dashed", linewidth = 0.6) +
  scale_x_log10(breaks = 10^(0:7), labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red", "gold")]),
                      breaks = c("Bayes: P(H0 | X) = 1/2", "5% test", "0.1% test")) +
  coord_cartesian(xlim = c(1, 1e7), ylim = c(0, 4.6), expand = FALSE) +
  labs(x = "sample size n", y = "critical value of |z|", title = "Rejection thresholds") +
  theme_blog() + theme(legend.position = "bottom", legend.direction = "vertical")
save_fig(p1 + p2, here, "lindley", width = 7.6, height = 4.2)
