# Linear mixed model as a hierarchical model: partial pooling of group means by the BLUP in a
# random-intercept model, and the sampling distributions of the ML and REML estimators of the
# random-effect variance in a balanced one-way layout.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1974)
mu <- 10; tau <- 1; sigma <- 2
nj <- c(2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 40); J <- length(nj)
b <- tau * rnorm(J)
ybar <- sapply(1:J, function(j) mu + b[j] + mean(sigma * rnorm(nj[j])))
w <- 1 / (tau^2 + sigma^2 / nj)
mu_hat <- sum(w * ybar) / sum(w)
lam <- nj * tau^2 / (nj * tau^2 + sigma^2)
blup <- mu_hat + lam * (ybar - mu_hat)
d1 <- rbind(data.frame(n = nj, y = ybar, what = "group mean"),
            data.frame(n = nj, y = blup, what = "BLUP"),
            data.frame(n = nj, y = mu + b, what = "true group mean"))
d1$what <- factor(d1$what, levels = c("group mean", "BLUP", "true group mean"))
p1 <- ggplot() +
  geom_hline(yintercept = mu_hat, colour = PALETTE[["grey"]], linetype = "dashed") +
  geom_segment(data = data.frame(n = nj, y0 = ybar, y1 = blup), aes(x = n, xend = n, y = y0, yend = y1),
               colour = PALETTE[["grey"]], linewidth = 0.35, arrow = arrow(length = unit(0.12, "cm"), type = "closed")) +
  geom_point(data = d1, aes(n, y, shape = what, colour = what), size = 1.9, stroke = 0.7) +
  scale_shape_manual(values = c(1, 16, 4)) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "blue", "red")])) +
  scale_x_log10(breaks = c(2, 5, 10, 20, 40)) +
  labs(x = expression("group size" ~ n[j] ~ "(log scale)"), y = "estimate of group mean", title = "Partial pooling of group means") +
  theme_blog() + theme(legend.position = c(0.78, 0.15), legend.justification = c(0.5, 0.5))

Jb <- 6; nb <- 5; tau2b <- 1; sig2b <- 4; reps <- 20000
set.seed(1977)
est <- t(replicate(reps, {
  y <- matrix(rep(sqrt(tau2b) * rnorm(Jb), each = nb) + sqrt(sig2b) * rnorm(Jb * nb), nb, Jb)
  gmeans <- colMeans(y); gm <- mean(y)
  msb <- nb * sum((gmeans - gm)^2) / (Jb - 1)
  msw <- sum(sweep(y, 2, gmeans)^2) / (Jb * (nb - 1))
  c(reml = max((msb - msw) / nb, 0), ml = max(((1 - 1 / Jb) * msb - msw) / nb, 0))
}))
lab <- c(sprintf("REML: mean %.2f, P(0) = %.2f", mean(est[, "reml"]), mean(est[, "reml"] == 0)),
         sprintf("ML: mean %.2f, P(0) = %.2f", mean(est[, "ml"]), mean(est[, "ml"] == 0)))
d2 <- rbind(data.frame(v = est[, "reml"], what = lab[1]), data.frame(v = est[, "ml"], what = lab[2]))
d2$what <- factor(d2$what, levels = lab)
means <- data.frame(m = c(mean(est[, "reml"]), mean(est[, "ml"])), what = factor(lab, levels = lab))
p2 <- ggplot(d2, aes(v, colour = what)) +
  stat_ecdf(geom = "step", linewidth = 0.7, pad = FALSE) +
  geom_vline(data = means, aes(xintercept = m, colour = what), linetype = "dashed", linewidth = 0.4, show.legend = FALSE) +
  geom_vline(xintercept = tau2b, colour = PALETTE[["ink"]], linewidth = 0.4) +
  annotate("text", x = tau2b + 0.12, y = 0.42, label = "'true'~tau^2 == 1", parse = TRUE, hjust = 0, size = 3.2) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  coord_cartesian(xlim = c(0, 6), ylim = c(0, 1.02), expand = FALSE) +
  labs(x = expression("estimate of" ~ tau^2), y = "distribution function", title = "ML and REML, J = 6, n = 5") +
  theme_blog() + theme(legend.position = c(0.62, 0.15), legend.justification = c(0.5, 0.5))
save_fig(p1 + p2 + plot_layout(widths = c(1.15, 1)), here, "pooling-reml", width = 7.8, height = 3.5)
