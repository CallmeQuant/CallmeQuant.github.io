# (a) Tail bounds for a standard normal variable: Markov (applied to |X|), Chebyshev and Chernoff
# against the exact two-sided tail. (b) Johnson-Lindenstrauss: squared-distance ratios after a
# Gaussian random projection of N = 100 points from R^5000 to R^k.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1916)
a <- seq(0.5, 5, length.out = 400)
lv <- c("exact", "Markov", "Chebyshev", "Chernoff")
tails <- rbind(
  data.frame(a = a, p = 2 * pnorm(a, lower.tail = FALSE), b = "exact"),
  data.frame(a = a, p = sqrt(2 / pi) / a, b = "Markov"),
  data.frame(a = a, p = 1 / a^2, b = "Chebyshev"),
  data.frame(a = a, p = 2 * exp(-a^2 / 2), b = "Chernoff"))
tails$b <- factor(tails$b, levels = lv)
pa <- ggplot(tails, aes(a, p, colour = b, linetype = b)) + geom_line(linewidth = 0.8) +
  scale_y_log10(limits = c(1e-7, 2), labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "gold", "red", "blue")])) +
  scale_linetype_manual(values = c("solid", "dotted", "dashed", "solid")) +
  labs(x = "a", y = NULL, title = "(a) tails of a standard normal") +
  theme_blog() + theme(legend.position = c(0.25, 0.25))

N <- 100; d <- 5000
P <- matrix(rnorm(N * d), N, d) * runif(N, 0.5, 2)
sqdist <- function(Y) { D <- as.matrix(dist(Y))^2; D[upper.tri(D)] }
D0 <- sqdist(P)
ratios <- do.call(rbind, lapply(c(25, 100, 400), function(k) {
  A <- matrix(rnorm(k * d), k, d) / sqrt(k)
  data.frame(r = sqdist(P %*% t(A)) / D0, k = paste0("k = ", k))
}))
ratios$k <- factor(ratios$k, levels = c("k = 25", "k = 100", "k = 400"))
pb <- ggplot(ratios, aes(r, fill = k, colour = k)) +
  geom_histogram(aes(y = after_stat(density)), breaks = seq(0.2, 2.0, length.out = 121),
                 position = "identity", alpha = 0.25, linewidth = 0.15) +
  geom_vline(xintercept = 1, linetype = "dotted", colour = PALETTE[["ink"]]) +
  scale_fill_manual(values = unname(PALETTE[c("gold", "red", "blue")])) +
  scale_colour_manual(values = unname(PALETTE[c("gold", "red", "blue")])) +
  labs(x = "squared-distance ratio after projection", y = "density",
       title = "(b) random projection, d = 5000, N = 100") +
  theme_blog() + theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))
save_fig(pa + pb, here, "tails-and-projection", width = 8, height = 3.6)
