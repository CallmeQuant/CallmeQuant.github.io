# Effective sample size for AR(1) sequences: variance inflation and estimation of tau.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1992)
ar1 <- function(phi, n, m = 1) {        # m stationary paths (rows), unit marginal variance
  e <- matrix(rnorm(m * n) * sqrt(1 - phi^2), m)
  e[, 1] <- rnorm(m)
  t(apply(e, 1, function(z) as.numeric(stats::filter(z, phi, method = "recursive"))))
}
inflation <- function(phi, n) (1 + phi) / (1 - phi) - 2 * phi * (1 - phi^n) / (n * (1 - phi)^2)
phis <- c(-0.5, 0.5, 0.9); labs_phi <- paste0("phi == ", phis)
cols <- unname(PALETTE[c("green", "blue", "red")])

dp <- do.call(rbind, lapply(1:3, function(j) data.frame(t = 1:200, x = ar1(phis[j], 200)[1, ] + 5 * (3 - j), k = labs_phi[j])))
dp$k <- factor(dp$k, levels = labs_phi)
p1 <- ggplot(dp, aes(t, x, colour = k)) + geom_line(linewidth = 0.35) +
  annotate("text", x = 205, y = 5 * (2:0), label = labs_phi, parse = TRUE, colour = cols, hjust = 0, size = 3.2) +
  scale_colour_manual(values = cols, guide = "none") + coord_cartesian(xlim = c(0, 235)) +
  labs(x = expression(t), y = NULL, title = "Stationary AR(1) paths, unit variance") +
  theme_blog() + theme(axis.text.y = element_blank(), panel.grid.major = element_blank())

ns <- unique(round(10^seq(0, 3.3, length.out = 60)))
dc <- do.call(rbind, lapply(1:3, function(j) data.frame(n = ns, v = inflation(phis[j], ns), k = labs_phi[j])))
ds <- do.call(rbind, lapply(1:3, function(j) do.call(rbind, lapply(c(5, 20, 100, 500), function(n)
  data.frame(n = n, v = n * var(rowMeans(ar1(phis[j], n, 4000))), k = labs_phi[j])))))
dc$k <- factor(dc$k, levels = labs_phi); ds$k <- factor(ds$k, levels = labs_phi)
p2 <- ggplot() +
  annotate("segment", x = 20, xend = 2000, y = (1 + phis) / (1 - phis), yend = (1 + phis) / (1 - phis),
           colour = cols, linetype = "dotted", linewidth = 0.3) +
  geom_hline(yintercept = 1, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_line(data = dc, aes(n, v, colour = k), linewidth = 0.8) +
  geom_point(data = ds, aes(n, v, colour = k), shape = 21, fill = "white", size = 1.6, stroke = 0.6) +
  scale_colour_manual(values = cols, labels = scales::label_parse()) + scale_x_log10() + scale_y_log10() +
  coord_cartesian(xlim = c(1, 2000), ylim = c(0.2, 40)) +
  labs(x = expression(n), y = expression(n ~ Var(bar(X)[n]) / gamma[0]), title = expression("Variance inflation approaches" ~ tau == (1 + phi) / (1 - phi))) +
  theme_blog() + theme(legend.position = "top")
save_fig(p1 + p2 + plot_layout(widths = c(1.1, 1)), here, "variance-inflation", width = 9.4, height = 3.4)

acov <- function(x) { n <- length(x); as.numeric(acf(x, lag.max = n - 1, type = "covariance", demean = TRUE, plot = FALSE)$acf) }
tau_geyer <- function(x) {
  g <- acov(x); n <- length(g); m <- floor(n / 2)
  G <- g[seq(1, 2 * m, 2)] + g[seq(2, 2 * m, 2)]
  M <- if (any(G <= 0)) which(G <= 0)[1] - 1 else m
  (-g[1] + 2 * sum(G[seq_len(M)])) / g[1]
}
tau_batch <- function(x) {
  n <- length(x); m <- floor(sqrt(n)); b <- n %/% m
  bm <- colMeans(matrix(x[1:(b * m)], nrow = m))
  m * var(bm) / (mean((x - mean(x))^2))
}
phi <- 0.9; n <- 5000; tau_true <- (1 + phi) / (1 - phi)
x <- ar1(phi, n)[1, ]
g <- acov(x)
partial <- (g[1] + 2 * cumsum(g[-1])) / g[1]
q1 <- ggplot(data.frame(K = 1:(n - 1), s = partial), aes(K, s)) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_hline(yintercept = tau_true, colour = PALETTE[["red"]], linetype = "dashed") +
  geom_line(colour = PALETTE[["blue"]], linewidth = 0.4) + scale_x_log10() +
  annotate("text", x = 1.2, y = tau_true + 2, label = "true ~ tau == 19", parse = TRUE, hjust = 0, colour = PALETTE[["red"]], size = 3.2) +
  labs(x = expression("truncation lag" ~ K), y = expression(1 + 2 * sum(hat(rho)[k], k <= K, "")), title = expression("One chain," ~ phi == 0.9 * "," ~ n == 5000)) +
  theme_blog()
X <- ar1(phi, n, 2000)
de <- rbind(data.frame(e = n / apply(X, 1, tau_batch), k = "batch means"),
            data.frame(e = n / apply(X, 1, tau_geyer), k = "initial positive sequence"))
q2 <- ggplot(de, aes(e, fill = k)) +
  geom_histogram(breaks = seq(100, 600, length.out = 51), position = "identity", alpha = 0.55) +
  geom_vline(xintercept = n / tau_true, colour = PALETTE[["red"]], linetype = "dashed") +
  scale_fill_manual(values = unname(PALETTE[c("gold", "blue")])) +
  labs(x = expression("estimated" ~ n[eff]), y = "chains", title = expression("Estimates from 2,000 chains (dashed:" ~ n / tau * ")")) +
  theme_blog() + theme(legend.position = c(1, 0.62), legend.justification = c(1, 0.5), legend.text = element_text(size = 8))
save_fig(q1 + q2 + plot_layout(widths = c(1.1, 1)), here, "estimating-tau", width = 9.4, height = 3.4)
