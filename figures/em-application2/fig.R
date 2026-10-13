# EM versus stochastic EM (SEM) for a two-component univariate Gaussian mixture.
# (a) trajectories of the ordered component means; (b) SEM iterates of the larger mean after burn-in.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(113)
n <- 400
z <- runif(n) < 0.35
x <- ifelse(z, rnorm(n, 2, 0.8), rnorm(n, -1, 1.2))
resp <- function(w, mu, s2) {
  lp <- sapply(1:2, function(k) log(w[k]) - 0.5 * log(2 * pi * s2[k]) - 0.5 * (x - mu[k])^2 / s2[k])
  m <- apply(lp, 1, max); e <- exp(lp - m); e / rowSums(e)
}
m_step <- function(g) {
  Nk <- colSums(g); mu <- colSums(g * x) / Nk
  s2 <- colSums(g * outer(x, mu, "-")^2) / Nk
  list(w = Nk / n, mu = mu, s2 = s2)
}
T <- 3000; burn <- 300
start <- list(w = c(0.5, 0.5), mu = c(-0.01, 0.01), s2 = c(3, 3))

p <- start; em <- matrix(NA, T + 1, 2); em[1, ] <- sort(p$mu)
for (t in 1:T) { p <- m_step(resp(p$w, p$mu, p$s2)); em[t + 1, ] <- sort(p$mu) }
p <- start; sem <- matrix(NA, T + 1, 2); sem[1, ] <- sort(p$mu)
for (t in 1:T) {
  g <- resp(p$w, p$mu, p$s2)
  repeat {                      # S-step; redraw if a component gets fewer than 2 points
    lab <- 1 + (runif(n) > g[, 1])
    oh <- cbind(lab == 1, lab == 2) * 1
    if (min(colSums(oh)) >= 2) break
  }
  p <- m_step(oh)
  sem[t + 1, ] <- sort(p$mu)
}

k <- 500
tr <- rbind(
  data.frame(t = 0:k, mu = sem[1:(k + 1), 1], what = "SEM, smaller mean"),
  data.frame(t = 0:k, mu = sem[1:(k + 1), 2], what = "SEM, larger mean"),
  data.frame(t = 0:k, mu = em[1:(k + 1), 1], what = "EM, smaller mean"),
  data.frame(t = 0:k, mu = em[1:(k + 1), 2], what = "EM, larger mean"))
tr$what <- factor(tr$what, levels = unique(tr$what))
pa <- ggplot(tr, aes(t, mu, colour = what, linetype = what, linewidth = what)) + geom_line() +
  scale_colour_manual(values = unname(c(PALETTE["blue"], PALETTE["red"], PALETTE["ink"], PALETTE["ink"]))) +
  scale_linetype_manual(values = c("solid", "solid", "solid", "dashed")) +
  scale_linewidth_manual(values = c(0.35, 0.35, 0.7, 0.7)) +
  labs(x = "iteration t", y = NULL, title = "(a) component means") +
  theme_blog() + theme(legend.position = "bottom") + guides(colour = guide_legend(nrow = 2))
draws <- sem[(burn + 2):(T + 1), 2]
pb <- ggplot(data.frame(m = draws), aes(m)) +
  geom_histogram(aes(y = after_stat(density)), bins = 40, fill = PALETTE[["red"]], alpha = 0.35) +
  geom_vline(data = data.frame(v = c(em[T + 1, 2], mean(draws)), lab = c("EM limit", "SEM average")),
             aes(xintercept = v, linetype = lab, colour = lab), linewidth = 0.7, key_glyph = "path") +
  scale_linetype_manual(values = c("EM limit" = "solid", "SEM average" = "dashed")) +
  scale_colour_manual(values = c("EM limit" = PALETTE[["ink"]], "SEM average" = PALETTE[["red"]])) +
  labs(x = "larger mean", y = "density", title = "(b) SEM iterates, t > 300") +
  theme_blog() + theme(legend.position = "bottom", legend.title = element_blank())
save_fig(pa + pb + plot_layout(widths = c(1.5, 1)), here, "em-vs-sem", width = 8, height = 3.9)
