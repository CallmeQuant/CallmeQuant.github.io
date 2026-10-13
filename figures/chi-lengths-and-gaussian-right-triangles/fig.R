# Chi lengths of projections of a standard Gaussian vector; chi densities; t statistic as a cotangent.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1908)
n <- 10; k <- 3; M <- 1500
Z <- matrix(rnorm(M * n), M)
Q <- qr.Q(qr(matrix(rnorm(n * n), n)))
Y <- Z %*% Q
a <- sqrt(rowSums(Y[, 1:k]^2)); b <- sqrt(rowSums(Y[, (k + 1):n]^2))
dchi <- function(r, k) exp((k - 1) * log(r) - r^2 / 2 - (k / 2 - 1) * log(2) - lgamma(k / 2))
qchi <- function(p, k) sqrt(qchisq(p, k))
th <- seq(0, pi / 2, length.out = 200)
arcs <- do.call(rbind, lapply(c(0.1, 0.5, 0.9), function(q)
  data.frame(x = qchi(q, n) * cos(th), y = qchi(q, n) * sin(th), q = factor(q))))
i <- 8
tri <- data.frame(x = c(0, a[i], a[i], 0), y = c(0, 0, b[i], 0))
p1 <- ggplot() +
  geom_point(aes(a, b), colour = PALETTE[["grey"]], alpha = 0.5, size = 0.35) +
  geom_path(data = arcs, aes(x, y, group = q, linetype = q), colour = PALETTE[["ink"]], linewidth = 0.35, show.legend = FALSE) +
  scale_linetype_manual(values = c("dotted", "dashed", "dotted")) +
  geom_path(data = tri, aes(x, y), colour = PALETTE[["red"]], linewidth = 0.8) +
  annotate("label", x = a[i] / 2, y = -0.3, label = "'||' * PZ * '||'", parse = TRUE, colour = PALETTE[["red"]], size = 3, label.size = 0) +
  annotate("label", x = a[i] + 0.1, y = b[i] / 2, label = "'||' * (I - P) * Z * '||'", parse = TRUE, colour = PALETTE[["red"]], size = 3,
           hjust = 0, label.size = 0) +
  coord_equal(xlim = c(0, 5), ylim = c(-0.55, 5.5)) +
  labs(x = expression("leg in" ~ L ~ (chi[3])), y = expression("leg in" ~ L^symbol("\136") ~ (chi[7])), title = expression(n == 10 * "," ~ dim ~ L == 3)) + theme_blog()

r <- seq(0, 9, length.out = 600); ks <- c(1, 2, 3, 10, 50)
dd <- do.call(rbind, lapply(ks, function(kk) data.frame(r = r, d = dchi(r, kk), k = factor(paste0("chi[", kk, "]"), levels = paste0("chi[", ks, "]")))))
p2 <- ggplot(dd, aes(r, d, colour = k)) + geom_line(linewidth = 0.8) +
  geom_vline(data = data.frame(v = sqrt(ks), k = factor(paste0("chi[", ks, "]"), levels = paste0("chi[", ks, "]"))),
             aes(xintercept = v, colour = k), linetype = "dotted", linewidth = 0.3, show.legend = FALSE) +
  scale_colour_manual(values = CYCLE[1:5], labels = scales::label_parse()) + coord_cartesian(xlim = c(0, 9), ylim = c(0, 0.9)) +
  labs(x = "r", y = NULL, title = expression("Densities of" ~ chi[k] * "; dotted at" ~ sqrt(k))) +
  theme_blog() + theme(legend.position = c(0.42, 1), legend.justification = c(0, 1))

m <- 6; S <- 200000
X <- matrix(rnorm(S * m, mean = 2, sd = 3), S)
Zc <- (X - 2) / 3
signed_leg <- sqrt(m) * rowMeans(Zc)
other_leg <- sqrt(rowSums((Zc - rowMeans(Zc))^2))
theta <- atan2(other_leg, signed_leg)
Tst <- sqrt(m - 1) / tan(theta)
x <- seq(-6, 6, length.out = 400)
p3 <- ggplot() +
  geom_histogram(aes(Tst[abs(Tst) < 6], after_stat(density)), breaks = seq(-6, 6, length.out = 81),
                 fill = PALETTE[["grey"]], alpha = 0.45, colour = "white", linewidth = 0.15) +
  geom_line(aes(x, dt(x, m - 1) / mean(abs(Tst) < 6)), colour = PALETTE[["blue"]], linewidth = 0.8) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(0, 0.45)) +
  labs(x = expression(sqrt(n - 1) ~ cot ~ vartheta * "," ~ n == 6), y = NULL, title = expression("The" ~ italic(t) ~ "statistic as a cotangent (blue:" ~ t[5] * ")")) + theme(plot.title = element_text(size = 10)) + theme_blog()
save_fig(p1 + p2 + p3 + plot_layout(widths = c(1, 1.05, 1.05)), here, "chi-triangles", width = 10, height = 3.4)
