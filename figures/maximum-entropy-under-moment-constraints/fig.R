# Maximum entropy pmfs on {0,...,10} with a mean constraint.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(2026)
K <- 10; k <- 0:K
maxent <- function(m) {
  gap <- function(th) { w <- exp(th * k - max(th * k)); sum(w * k) / sum(w) - m }
  th <- uniroot(gap, c(-60, 60), tol = 1e-12)$root
  w <- exp(th * k - max(th * k)); list(p = w / sum(w), th = th)
}
H <- function(p) -sum(ifelse(p > 0, p * log(p), 0))

means <- c(1, 3, 5, 7.5)
d1 <- do.call(rbind, lapply(means, function(m) {
  r <- maxent(m); data.frame(k = k, p = r$p, lab = sprintf("list(m == %g, theta == %.2f)", m, r$th))
}))
d1$lab <- factor(d1$lab, levels = unique(d1$lab))
p1 <- ggplot(d1, aes(k, p, fill = lab)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.8) +
  scale_fill_manual(values = CYCLE[1:4], labels = function(x) parse(text = x)) +
  scale_x_continuous(breaks = k) + coord_cartesian(ylim = c(0, 0.68)) +
  labs(x = "k", y = expression(p[theta](k)), title = expression("Maximizers " * p[theta](k) %prop% e^{theta * k})) +
  theme_blog() + theme(legend.position = c(0.5, 0.88), legend.justification = c(0.5, 0.5),
                       legend.key.size = unit(0.7, "lines"), legend.text = element_text(size = 7.5),
                       panel.grid.major.x = element_blank()) +
  guides(fill = guide_legend(ncol = 2))

G <- matrix(rexp(6000 * (K + 1)), ncol = K + 1); P <- G / rowSums(G)   # Dirichlet(1,...,1)
dr <- data.frame(m = P %*% k, h = apply(P, 1, H))
ms <- seq(0.05, K - 0.05, length.out = 300)
dm <- data.frame(m = ms, h = sapply(ms, function(m) H(maxent(m)$p)), what = "maximum entropy")
db <- data.frame(m = ms, h = sapply(ms, function(m) H(dbinom(k, K, m / K))), what = "Binomial(10, m/10)")
p2 <- ggplot() +
  geom_point(data = dr, aes(m, h), colour = PALETTE[["grey"]], alpha = 0.35, size = 0.3) +
  geom_hline(yintercept = log(K + 1), linetype = "dotted", colour = PALETTE[["ink"]], linewidth = 0.3) +
  annotate("text", x = 0.2, y = log(K + 1) + 0.08, label = "log 11", hjust = 0, size = 3.2) +
  geom_line(data = rbind(dm, db), aes(m, h, colour = what, linetype = what), linewidth = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  coord_cartesian(xlim = c(0, K), ylim = c(0, 2.75)) +
  labs(x = "mean m", y = "entropy (nats)", title = "Entropy against mean (grey: random pmfs)") +
  theme_blog() + theme(legend.position = c(0.5, 0.15), legend.justification = c(0.5, 0.5))
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.1)), here, "maxent-discrete", width = 8.2, height = 3.5)
