# Riemann sums for int W dW with different evaluation points; geometric Brownian motion.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1944)
L <- 18; K <- 2^L
W <- c(0, cumsum(rnorm(K, sd = sqrt(1 / K))))
levels <- 1:(L - 1)
rows <- lapply(levels, function(l) {
  step <- 2^(L - l); idx <- seq(1, K + 1, by = step)
  Wg <- W[idx]; dW <- diff(Wg); Wm <- W[idx[-length(idx)] + step / 2]
  data.frame(n = 2^l, left = sum(Wg[-length(Wg)] * dW), mid = sum(Wm * dW), right = sum(Wg[-1] * dW))
})
d <- do.call(rbind, rows)
w1 <- W[K + 1]
lab <- c(right = "right endpoint: limit (W1^2 + 1)/2", mid = "midpoint: limit W1^2/2",
         left = "left endpoint: limit (W1^2 - 1)/2")
dl <- rbind(data.frame(n = d$n, s = d$right, k = lab["right"]), data.frame(n = d$n, s = d$mid, k = lab["mid"]),
            data.frame(n = d$n, s = d$left, k = lab["left"]))
dl$k <- factor(dl$k, levels = lab)
lims <- data.frame(k = factor(lab, levels = lab), y = c(0.5 * (w1^2 + 1), 0.5 * w1^2, 0.5 * (w1^2 - 1)))
cols <- unname(PALETTE[c("red", "green", "blue")])
p1 <- ggplot(dl, aes(n, s, colour = k)) +
  geom_hline(data = lims, aes(yintercept = y, colour = k), linetype = "dashed", linewidth = 0.3) +
  geom_line(linewidth = 0.4) + geom_point(size = 1) +
  scale_x_continuous(trans = "log2", breaks = 2^seq(1, 17, 4), labels = scales::label_math(2^.x, format = log2)) +
  scale_colour_manual(values = cols, labels = expression("right endpoint" %->% (W[1]^2 + 1) / 2,
                                                      "midpoint" %->% W[1]^2 / 2,
                                                      "left endpoint" %->% (W[1]^2 - 1) / 2)) +
  coord_cartesian(ylim = c(min(lims$y, dl$s) - 0.35, max(lims$y, dl$s) + 0.95)) +
  guides(colour = guide_legend(ncol = 1)) +
  labs(x = "number of intervals n", y = NULL, title = expression("Sums" ~ sum(W[s[i]] * (W[t[i]] - W[t[i - 1]]), i) * ", one path")) +
  theme_blog() + theme(legend.position = c(0.5, 0.88), legend.justification = c(0.5, 0.5))

M <- 3000; nint <- 500
dWs <- matrix(rnorm(M * nint, sd = sqrt(1 / nint)), nrow = M)
Ws <- t(apply(dWs, 1, cumsum))
left <- rowSums(cbind(0, Ws[, -nint]) * dWs)
x <- seq(-3.5, 3.5, length.out = 300)
p2 <- ggplot() +
  geom_point(aes(Ws[, nint], left), colour = PALETTE[["blue"]], alpha = 0.35, size = 0.4) +
  geom_line(aes(x, 0.5 * x^2), colour = PALETTE[["grey"]], linewidth = 0.6) +
  geom_line(aes(x, 0.5 * (x^2 - 1)), colour = PALETTE[["ink"]], linewidth = 0.6, linetype = "dashed") +
  annotate("text", x = -0.9, y = 5.3, hjust = 0, size = 3, label = "grey: w^2/2,  dashed: (w^2 - 1)/2") +
  coord_cartesian(xlim = c(-3.5, 3.5), ylim = c(-1, 5.5)) +
  labs(x = expression(W[1]), y = NULL, title = "Left sums (n = 500) across 3,000 paths") + theme_blog()
save_fig(p1 + p2 + plot_layout(widths = c(1.15, 1)), here, "riemann-sums", width = 8.8, height = 3.4)

mu <- 0.1; sigma <- 0.6; T <- 10; N <- 1000; P <- 2000
tt <- seq(0, T, length.out = N + 1)
Wp <- cbind(0, t(apply(matrix(rnorm(P * N, sd = sqrt(T / N)), nrow = P), 1, cumsum)))
S <- exp(sweep(sigma * Wp, 2, (mu - sigma^2 / 2) * tt, "+"))
q <- apply(S, 2, quantile, probs = c(0.1, 0.9))
paths <- data.frame(t = rep(tt, each = 25), s = as.vector(S[1:25, ]), id = rep(1:25, N + 1))
curves <- rbind(
  data.frame(t = tt, s = exp(mu * tt), k = "mean exp(mu t)"),
  data.frame(t = tt, s = exp((mu - sigma^2 / 2) * tt), k = "median exp((mu - sigma^2/2) t)"),
  data.frame(t = tt, s = colMeans(S), k = "sample mean of 2,000 paths"))
curves$k <- factor(curves$k, levels = unique(curves$k))
qd <- rbind(data.frame(t = tt, s = q[1, ], w = "q10"), data.frame(t = tt, s = q[2, ], w = "q90"))
labs_k <- expression("mean" ~ e^{mu * t}, "median" ~ e^{(mu - sigma^2 / 2) * t}, "sample mean of 2,000 paths")
g <- ggplot() +
  geom_line(data = paths, aes(t, s, group = id), colour = PALETTE[["grey"]], linewidth = 0.25, alpha = 0.7) +
  geom_line(data = qd, aes(t, s, group = w), colour = PALETTE[["violet"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(data = curves, aes(t, s, colour = k, linetype = k, linewidth = k)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "red")]), labels = labs_k) +
  scale_linetype_manual(values = c("solid", "solid", "dashed"), labels = labs_k) +
  scale_linewidth_manual(values = c(0.9, 0.9, 0.4), labels = labs_k) +
  scale_y_log10(limits = c(1e-3, 1e2), breaks = 10^(-3:2), labels = c("0.001", "0.01", "0.1", "1", "10", "100")) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(x = "t", y = expression(S[t] ~ "(log scale)"),
       title = expression(dS[t] == mu * S[t] * dt + sigma * S[t] * dW[t] * "," ~ mu == 0.1 * "," ~ sigma == 0.6 * "," ~ S[0] == 1),
       subtitle = "dotted: 10% and 90% quantiles across 2,000 paths") +
  theme_blog() + theme(legend.position = "bottom", legend.key.size = unit(0.8, "lines"),
                       plot.margin = margin(5, 14, 5, 5))
save_fig(g, here, "gbm", width = 6.8, height = 4.2)
