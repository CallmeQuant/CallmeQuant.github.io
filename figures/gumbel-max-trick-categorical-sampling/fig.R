# Gumbel-max sampling of a categorical distribution and the Gumbel-softmax relaxation.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(7)
rgumbel <- function(n) -log(-log(runif(n)))

# ---- Figure 1
a <- c(1.2, 0.3, -0.4, 0.9, -1.5, 0.0); K <- length(a)
pi_k <- exp(a) / sum(exp(a)); logS <- log(sum(exp(a)))
N <- 200000
Y <- matrix(rgumbel(N * K), N) + rep(a, each = N)
I <- max.col(Y, ties.method = "first"); M <- Y[cbind(seq_len(N), I)]
d1 <- data.frame(k = 1:K, pi = pi_k, freq = tabulate(I, K) / N)
p1 <- ggplot(d1, aes(k)) +
  geom_col(aes(y = pi, fill = "pi"), width = 0.6, colour = PALETTE[["grey"]]) +
  geom_point(aes(y = freq, colour = "freq"), size = 2.5) +
  scale_fill_manual(values = c(pi = PALETTE[["light"]]), labels = expression(pi[k] == e^{a[k]} / sum(e^{a[j]}, j))) +
  scale_colour_manual(values = c(freq = PALETTE[["blue"]]), labels = expression("frequency of" ~ arg ~ max[j] ~ (a[j] + G[j]) == k)) +
  scale_x_continuous(breaks = 1:K) + coord_cartesian(ylim = c(0, 0.56)) +
  labs(x = "category k", y = "probability", title = expression(2 %*% 10^5 ~ "draws," ~ K == 6)) +
  theme_blog() + theme(legend.position = c(0.98, 0.99), legend.justification = c(1, 1),
                       legend.box.just = "right", legend.spacing.y = unit(0, "pt"), legend.text = element_text(size = 8))

x <- seq(-2, 8, length.out = 400)
dens <- data.frame(x = x, y = exp(-(x - logS) - exp(-(x - logS))))
lab_of <- function(j) sprintf("M given I = %d (prob. %.3f)", j, pi_k[j])
dm <- rbind(data.frame(M = M[I == 1], g = lab_of(1)), data.frame(M = M[I == 5], g = lab_of(5)))
p2 <- ggplot(dm, aes(M, colour = g)) +
  geom_step(aes(y = after_stat(density)), stat = "bin", breaks = seq(-2, 8, length.out = 70), linewidth = 0.5) +
  geom_line(data = dens, aes(x, y, colour = "Gumbel(log S) density"), linetype = 2, linewidth = 0.45) +
  scale_colour_manual(values = setNames(unname(PALETTE[c("blue", "red", "ink")]),
                                        c(lab_of(1), lab_of(5), "Gumbel(log S) density")),
                      breaks = c(lab_of(1), lab_of(5), "Gumbel(log S) density")) +
  coord_cartesian(xlim = c(-2, 8), ylim = c(0, 0.62)) +
  labs(x = expression(M == max[j] ~ (a[j] + G[j])), y = "density", title = "The maximum is independent of the argmax") +
  theme_blog() + theme(legend.position = c(0.98, 0.99), legend.justification = c(1, 1), legend.text = element_text(size = 8))
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.15)), here, "argmax-and-max", width = 7.8, height = 3.4)

# ---- Figure 2: Gumbel-softmax samples on the simplex
a3 <- log(c(0.5, 0.3, 0.2))
V <- rbind(c(0, 0), c(1, 0), c(0.5, sqrt(3) / 2))
taus <- c(0.1, 0.5, 1, 5); n <- 1500
pts <- do.call(rbind, lapply(taus, function(tau) {
  Yt <- (matrix(rgumbel(n * 3), n) + rep(a3, each = n)) / tau
  P <- exp(Yt - apply(Yt, 1, max)); P <- P / rowSums(P)
  xy <- P %*% V
  data.frame(x = xy[, 1], y = xy[, 2], win = factor(max.col(P)), tau = sprintf("tau == %s", tau))
}))
pts$tau <- factor(pts$tau, levels = sprintf("tau == %s", taus))
tri <- data.frame(x = V[c(1:3, 1), 1], y = V[c(1:3, 1), 2])
vl <- data.frame(x = V[, 1], y = V[, 2] + c(-0.08, -0.08, 0.06), lab = sprintf("e[%d]", 1:3))
p3 <- ggplot(pts, aes(x, y)) +
  geom_path(data = tri, colour = PALETTE[["grey"]], linewidth = 0.35) +
  geom_point(aes(colour = win), size = 0.35, alpha = 0.5) +
  geom_text(data = vl, aes(label = lab), parse = TRUE, family = "serif", size = 3.4) +
  facet_wrap(~tau, nrow = 1, labeller = label_parsed) +
  scale_colour_manual(values = CYCLE[1:3], guide = "none") +
  coord_equal(xlim = c(-0.08, 1.08), ylim = c(-0.16, 1)) +
  theme_void(base_family = "serif") + theme(strip.text = element_text(size = 11, margin = margin(b = 4)),
                                            plot.background = element_rect(fill = "white", colour = NA))
save_fig(p3, here, "gumbel-softmax", width = 8, height = 2.5)
