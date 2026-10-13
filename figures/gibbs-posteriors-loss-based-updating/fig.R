# Gibbs posteriors for the median under absolute-error loss with Gamma(2, 1) data: posteriors for
# several learning rates against a Gaussian-model posterior, and frequentist coverage of 95%
# credible intervals as a function of the learning rate.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

shape <- 2
med <- qgamma(0.5, shape)
omega_star <- 2 * dgamma(med, shape)
grid <- seq(0, 8, length.out = 3201); dg <- grid[2] - grid[1]

gibbs <- function(x, omegas) {
  loss <- colSums(abs(outer(x, grid, "-")))
  lw <- -outer(omegas, loss)
  w <- exp(lw - apply(lw, 1, max))
  w / (rowSums(w) * dg)
}

set.seed(31)
n <- 50
x <- rgamma(n, shape, 1)
omegas <- c(0.2, omega_star, 3)
post <- gibbs(x, omegas)
labs_w <- c("omega == 0.2", sprintf("omega == omega^'*' %%~~%% %.2f", omega_star), "omega == 3", "'Gaussian model'")
d1 <- rbind(do.call(rbind, lapply(1:3, function(k) data.frame(t = grid, dens = post[k, ], what = labs_w[k]))),
            data.frame(t = grid, dens = dnorm(grid, mean(x), sd(x) / sqrt(n)), what = labs_w[4]))
d1$what <- factor(d1$what, levels = labs_w)
ytop <- 1.12 * max(post)
p1 <- ggplot(d1, aes(t, dens, colour = what, linetype = what)) +
  geom_vline(xintercept = med, colour = PALETTE[["ink"]], linetype = "dotted", linewidth = 0.4) +
  geom_vline(xintercept = shape, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  annotate("text", x = med - 0.03, y = 0.96 * ytop, label = "median", hjust = 1, size = 3) +
  annotate("text", x = shape + 0.03, y = 0.96 * ytop, label = "mean", hjust = 0, size = 3,
           colour = PALETTE[["grey"]]) +
  geom_line(linewidth = 0.75) +
  scale_colour_manual(values = c(CYCLE[1:3], PALETTE[["grey"]]), labels = scales::label_parse()) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed"), labels = scales::label_parse()) +
  coord_cartesian(xlim = c(0, 3.4), ylim = c(0, ytop), expand = FALSE) +
  labs(x = expression(theta), y = "posterior density", title = "One sample, n = 50") +
  theme_blog() + theme(legend.position = c(1, 0.62), legend.justification = c(1, 0.5), legend.text = element_text(size = 8.5),
                       legend.key.width = unit(1.2, "lines"))

reps <- 2000
omega_grid <- exp(seq(log(0.05), log(5), length.out = 31))
set.seed(32)
d2 <- do.call(rbind, lapply(c(50, 400), function(n) {
  hits <- rep(0, length(omega_grid))
  for (r in seq_len(reps)) {
    p <- gibbs(rgamma(n, shape, 1), omega_grid)
    cdf <- t(apply(p, 1, cumsum)) * dg
    lo <- grid[apply(cdf >= 0.025, 1, which.max)]
    hi <- grid[apply(cdf >= 0.975, 1, which.max)]
    hits <- hits + (lo <= med & med <= hi)
  }
  data.frame(omega = omega_grid, cov = hits / reps, what = paste("n =", n))
}))
og <- exp(seq(log(0.05), log(5), length.out = 400))
asym <- data.frame(omega = og, cov = 2 * pnorm(qnorm(0.975) * sqrt(omega_star / og)) - 1, what = "asymptotic")
p2 <- ggplot(d2, aes(omega, cov, colour = what)) +
  geom_hline(yintercept = 0.95, colour = PALETTE[["grey"]], linetype = "dashed", linewidth = 0.4) +
  geom_vline(xintercept = omega_star, colour = PALETTE[["ink"]], linetype = "dotted", linewidth = 0.4) +
  annotate("text", x = omega_star * 1.1, y = 0.08, label = "omega^'*'", parse = TRUE, hjust = 0, size = 3.4) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.1) +
  geom_line(data = asym, linewidth = 0.4) +
  scale_colour_manual(values = c(CYCLE[1:2], PALETTE[["ink"]]), breaks = c("n = 50", "n = 400", "asymptotic")) +
  scale_x_log10(breaks = c(0.05, 0.1, 0.2, 0.5, 1, 2, 5), labels = c("0.05", "0.1", "0.2", "0.5", "1", "2", "5")) +
  coord_cartesian(xlim = c(0.05, 5), ylim = c(0, 1.02), expand = FALSE) +
  labs(x = expression("learning rate" ~ omega ~ "(log scale)"), y = "coverage of 95% interval",
       title = "Frequentist coverage for the median") +
  theme_blog() + theme(legend.position = c(0.2, 0.2), legend.justification = c(0.5, 0.5))
save_fig(p1 + p2, here, "gibbs-median", width = 7.6, height = 3.4)
