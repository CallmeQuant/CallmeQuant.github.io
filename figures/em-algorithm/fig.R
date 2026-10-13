# EM for a two-component Poisson mixture on simulated counts (note: the EM algorithm).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(2023)
n <- 500; pi_true <- c(0.6, 0.4); lam_true <- c(2, 9)
z <- sample(1:2, n, replace = TRUE, prob = pi_true)
x <- rpois(n, lam_true[z])

logsumexp_rows <- function(M) { m <- apply(M, 1, max); m + log(rowSums(exp(M - m))) }
loglik <- function(pi, lam) {
  M <- sapply(1:2, function(k) log(pi[k]) + dpois(x, lam[k], log = TRUE))
  sum(logsumexp_rows(M))
}
em <- function(pi, lam, iters) {
  ll <- loglik(pi, lam)
  for (it in seq_len(iters)) {
    M <- sapply(1:2, function(k) log(pi[k]) + dpois(x, lam[k], log = TRUE))   # E-step
    r <- exp(M - logsumexp_rows(M))
    Nk <- colSums(r)                                                          # M-step
    pi <- Nk / n; lam <- colSums(r * x) / Nk
    ll <- c(ll, loglik(pi, lam))
  }
  list(pi = pi, lam = lam, ll = ll)
}

fit <- em(c(0.5, 0.5), c(1, 3), 2000)
ll_hat <- loglik(fit$pi, fit$lam)
k <- 0:(max(x) + 1)
freq <- data.frame(k = k, f = tabulate(x + 1, nbins = length(k)) / n)
curves <- rbind(
  data.frame(k = k, p = fit$pi[1] * dpois(k, fit$lam[1]), what = "component 1"),
  data.frame(k = k, p = fit$pi[2] * dpois(k, fit$lam[2]), what = "component 2"),
  data.frame(k = k, p = fit$pi[1] * dpois(k, fit$lam[1]) + fit$pi[2] * dpois(k, fit$lam[2]), what = "fitted mixture"))
cols <- c("component 1" = PALETTE[["blue"]], "component 2" = PALETTE[["red"]], "fitted mixture" = PALETTE[["ink"]])
pa <- ggplot() +
  geom_col(data = freq, aes(k, f), width = 0.85, fill = PALETTE[["light"]], colour = PALETTE[["grey"]], linewidth = 0.2) +
  geom_line(data = curves, aes(k, p, colour = what, linetype = what), linewidth = 0.6) +
  geom_point(data = subset(curves, what == "fitted mixture"), aes(k, p), colour = PALETTE[["ink"]], size = 0.8) +
  scale_colour_manual(values = cols) +
  scale_linetype_manual(values = c("dashed", "dashed", "solid")) +
  labs(x = "count x", y = "relative frequency", title = "(a) n = 500 counts and the EM fit") +
  theme_blog() + theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))

starts <- list(list(c(0.5, 0.5), c(1, 3), "start A"), list(c(0.9, 0.1), c(5, 6), "start B"),
               list(c(0.5, 0.5), c(4, 4), "start C (equal rates)"))
gaps <- do.call(rbind, lapply(starts, function(s) {
  r <- em(s[[1]], s[[2]], 30)
  data.frame(t = 0:30, gap = pmax(ll_hat - r$ll, 1e-12), start = s[[3]])
}))
gaps$start <- factor(gaps$start, levels = sapply(starts, `[[`, 3))
pb <- ggplot(gaps, aes(t, gap, colour = start)) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = CYCLE[1:3]) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-9, 3, 2)) +
  coord_cartesian(xlim = c(0, 30), ylim = c(1e-9, 3e3)) +
  labs(x = "iteration t", y = expression(italic(l)(hat(theta)) - italic(l)(theta[t])), title = "(b) log-likelihood gap") +
  theme_blog() + theme(legend.position = c(0.98, 0.55), legend.justification = c(1, 0.5))

save_fig(pa + pb, here, "poisson-mixture-em", width = 8.2, height = 3.4)
