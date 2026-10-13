# Cover's universal portfolio on a simulated two-stock market.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# Two stocks sharing a random-walk factor F, with a mean-reverting spread s.
set.seed(5)
n <- 750
F <- cumsum(c(0, rnorm(n, 0, 0.012)))
s <- numeric(n + 1)
for (t in 2:(n + 1)) s[t] <- 0.9 * s[t - 1] + rnorm(1, 0, 0.04)
logA <- F + s + 0.0002 * (0:n); logB <- F - s
X <- exp(cbind(diff(logA), diff(logB)))

w <- seq(0, 1, length.out = 4001)
logS <- rbind(0, apply(log(outer(X[, 1], w) + outer(X[, 2], 1 - w)), 2, cumsum))
mx <- apply(logS, 1, max)
W <- exp(logS - mx)
w_hat <- rowSums(sweep(W, 2, w, `*`)) / rowSums(W)
log_up <- c(0, cumsum(log(w_hat[1:n] * X[, 1] + (1 - w_hat[1:n]) * X[, 2])))
stopifnot(max(abs(log_up - (log(rowMeans(W)) + mx))) < 1e-6)        # Lemma 1, numerically
k_star <- which.max(logS[n + 1, ]); w_star <- w[k_star]
log_bh <- log(0.5 * exp(logA - logA[1]) + 0.5 * exp(logB - logB[1]))
tt <- 0:n

labs5 <- c("stock A", "stock B", "buy and hold, half in each",
           sprintf("best CRP in hindsight, w* = %.2f", w_star), "universal portfolio")
paths <- rbind(data.frame(t = tt, v = logA - logA[1], s = labs5[1]), data.frame(t = tt, v = logB - logB[1], s = labs5[2]),
               data.frame(t = tt, v = log_bh, s = labs5[3]), data.frame(t = tt, v = logS[, k_star], s = labs5[4]),
               data.frame(t = tt, v = log_up, s = labs5[5]))
paths$s <- factor(paths$s, labs5)
p1 <- ggplot(paths, aes(t, v, colour = s, linewidth = s)) + geom_hline(yintercept = 0, linewidth = 0.25) + geom_line() +
  scale_colour_manual(values = unname(PALETTE[c("grey", "violet", "gold", "green", "blue")])) +
  scale_linewidth_manual(values = c(0.3, 0.3, 0.45, 0.55, 0.65)) +
  coord_cartesian(xlim = c(0, n)) + scale_y_continuous(expand = expansion(mult = c(0.05, 0.55))) +
  labs(x = "period t", y = "log wealth", title = "Wealth over time (initial wealth 1)") +
  theme_blog() + theme(legend.position = c(0.01, 0.99), legend.justification = c(0, 1), legend.text = element_text(size = 7.5),
                       legend.key.height = unit(0.8, "lines"))
fin <- data.frame(w = w, S = exp(logS[n + 1, ]))
p2 <- ggplot(fin, aes(w, S)) +
  geom_line(aes(colour = "crp"), linewidth = 0.6) +
  geom_hline(aes(yintercept = exp(log_up[n + 1]), colour = "up"), linewidth = 0.55) +
  annotate("point", x = c(0, 1), y = fin$S[c(1, 4001)], colour = PALETTE[["ink"]], size = 1.6) +
  annotate("text", x = c(0.03, 0.97), y = fin$S[c(1, 4001)] - 0.12, label = c("all in B", "all in A"),
           hjust = c(0, 1), size = 3, family = "serif") +
  scale_colour_manual(values = c(crp = PALETTE[["green"]], up = PALETTE[["blue"]]),
                      labels = c(crp = expression(S[n](b) * ", CRP with weight" ~ w ~ "on A"),
                                 up = expression(hat(S)[n] == integral(S[n](b) * dw, 0, 1)))) +
  coord_cartesian(xlim = c(0, 1)) + scale_y_continuous(expand = expansion(mult = c(0.12, 0.45))) +
  labs(x = "weight w on stock A", y = sprintf("final wealth after n = %d periods", n), title = "Every constant rebalancing rule") +
  theme_blog() + theme(legend.position = c(0.5, 0.99), legend.justification = c(0.5, 1), legend.text = element_text(size = 7.5))
save_fig(p1 + p2 + plot_layout(widths = c(1.4, 1)), here, "wealth", width = 7.8, height = 3.5)

best_w <- w[apply(logS[2:(n + 1), ], 1, which.max)]
dw <- rbind(data.frame(t = 1:n, v = best_w, s = "best CRP weight up to time t"),
            data.frame(t = 1:n, v = w_hat[1:n], s = "universal portfolio weight"))
dw$s <- factor(dw$s, unique(dw$s))
p3 <- ggplot(dw, aes(t, v, colour = s, linewidth = s)) + geom_line() +
  geom_hline(yintercept = w_star, linetype = 2, linewidth = 0.3) +
  scale_colour_manual(values = unname(PALETTE[c("green", "blue")])) + scale_linewidth_manual(values = c(0.35, 0.6)) +
  coord_cartesian(xlim = c(0, n), ylim = c(0, 1)) +
  labs(x = "period t", y = "weight on stock A", title = "Portfolio weights") +
  theme_blog() + theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1), legend.text = element_text(size = 7.5))
dr <- rbind(data.frame(t = tt, v = mx - log_up, s = "log S*_t - log S^_t"),
            data.frame(t = tt, v = log(tt + 1) + 1, s = "bound (m - 1) log(t + 1) + 1, m = 2"))
dr$s <- factor(dr$s, unique(dr$s))
p4 <- ggplot(dr, aes(t, v, colour = s, linetype = s)) + geom_line(linewidth = 0.55) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]),
                      labels = expression(log ~ S[t]^"*" - log ~ hat(S)[t], "bound" ~ (m - 1) * log(t + 1) + 1 * "," ~ m == 2)) +
  scale_linetype_manual(values = c(1, 2),
                        labels = expression(log ~ S[t]^"*" - log ~ hat(S)[t], "bound" ~ (m - 1) * log(t + 1) + 1 * "," ~ m == 2)) +
  coord_cartesian(xlim = c(0, n), ylim = c(0, 8.5)) +
  labs(x = "period t", y = "regret (log scale of wealth)", title = "Regret against the best CRP") +
  theme_blog() + theme(legend.position = c(0.99, 0.45), legend.justification = c(1, 0.5), legend.text = element_text(size = 7.5))
save_fig(p3 + p4, here, "weights-regret", width = 7.8, height = 3.3)
