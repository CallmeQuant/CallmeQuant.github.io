# Rao-Blackwellization in a two-block Gibbs sampler for Student t_3 (data augmentation).
# Y | X ~ Gamma((nu+1)/2, rate (nu+X^2)/2), X | Y ~ N(0, 1/Y).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
set.seed(1990)
NU <- 3; N <- 500; A <- 3

gibbs <- function(n_iter, n_chains) {
  X <- matrix(0, n_chains, n_iter); Y <- X
  x <- rt(n_chains, NU)
  for (t in 1:n_iter) {
    y <- rgamma(n_chains, shape = (NU + 1) / 2, rate = (NU + x^2) / 2)
    x <- rnorm(n_chains) / sqrt(y)
    Y[, t] <- y; X[, t] <- x
  }
  list(X = X, Y = Y)
}

ch <- gibbs(N, 1); x <- ch$X[1, ]; y <- ch$Y[1, ]
grid <- seq(-8, 8, length.out = 801)
kd <- density(x, n = 801, from = -8, to = 8)
rb <- sapply(grid, function(g) mean(dnorm(g, sd = 1 / sqrt(y))))
dd <- rbind(data.frame(x = grid, d = dt(grid, NU), kind = "target t3 density"),
            data.frame(x = kd$x, d = kd$y, kind = "kernel estimate from X_t"),
            data.frame(x = grid, d = rb, kind = "Rao-Blackwell: mean of p(x | Y_t)"))
dd$kind <- factor(dd$kind, levels = unique(dd$kind))
p1 <- ggplot(subset(dd, d > 1e-5), aes(x, d, colour = kind, linetype = kind)) +
  geom_line(linewidth = 0.7) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "grey", "blue")]), labels = expression("target" ~ t[3] ~ "density", "kernel estimate from" ~ X[t], frac(1, N) * sum(pi(x ~ "|" ~ Y[t]), t))) +
  scale_linetype_manual(values = c("dashed", "solid", "solid"), labels = expression("target" ~ t[3] ~ "density", "kernel estimate from" ~ X[t], frac(1, N) * sum(pi(x ~ "|" ~ Y[t]), t))) +
  coord_cartesian(xlim = c(-8, 8), ylim = c(1e-4, 0.8)) +
  guides(colour = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1)) +
  labs(x = "x", y = "density (log scale)", title = sprintf("One chain, N = %d", N)) +
  theme_blog()

ch <- gibbs(N, 4000)
crude <- rowMeans(ch$X > A)
rbp <- rowMeans(pnorm(A * sqrt(ch$Y), lower.tail = FALSE))
truth <- pt(A, NU, lower.tail = FALSE)
hh <- rbind(data.frame(est = crude, kind = sprintf("crude: sd %.4f", sd(crude))),
            data.frame(est = rbp, kind = sprintf("Rao-Blackwell: sd %.4f", sd(rbp))))
hh$kind <- factor(hh$kind, levels = unique(hh$kind))
p2 <- ggplot(hh, aes(est, fill = kind)) +
  geom_histogram(breaks = ((0:36) - 0.5) / N, position = "identity", alpha = 0.6) +
  geom_vline(xintercept = truth, linetype = "dashed", colour = PALETTE[["ink"]], linewidth = 0.4) +
  scale_fill_manual(values = unname(PALETTE[c("grey", "blue")])) +
  guides(fill = guide_legend(ncol = 1)) +
  labs(x = sprintf("estimate of P(X > 3) = %.4f", truth), y = "number of chains",
       title = sprintf("4000 chains, N = %d each", N)) +
  theme_blog()
save_fig(p1 + p2 + plot_layout(widths = c(1.15, 1)), here, "rb-gibbs", width = 7.9, height = 3.9)
