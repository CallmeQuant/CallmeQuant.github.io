# Gambler's ruin paths and likelihood-ratio martingales with Ville's threshold.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1939)
a <- 6; b <- 4; n_show <- 14
run <- function() {
  s <- 0; path <- 0
  while (s > -a && s < b) { s <- s + sample(c(-1, 1), 1); path <- c(path, s) }
  path
}
d1 <- do.call(rbind, lapply(1:n_show, function(i) {
  p <- run(); data.frame(n = seq_along(p) - 1, s = p, id = i, top = p[length(p)] == b)
}))
ends <- do.call(rbind, lapply(split(d1, d1$id), function(x) x[nrow(x), ]))
p1 <- ggplot(d1, aes(n, s, group = id, colour = top)) +
  geom_hline(yintercept = c(-a, b), colour = PALETTE[["ink"]], linewidth = 0.4) +
  geom_step(alpha = 0.75, linewidth = 0.4) + geom_point(data = ends, size = 1.2) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")]), guide = "none") +
  scale_y_continuous(breaks = seq(-6, 4, 2)) +
  coord_cartesian(xlim = c(0, 80), ylim = c(-a - 0.8, b + 0.8)) +
  labs(x = "n", y = expression(S[n] ~ "stopped at" ~ tau), title = "Walks stopped at -6 or 4") + theme_blog()

theta <- 0.5; N <- 400; alpha <- 0.05; n <- 1:N
log10_lr <- function(mean, m) {
  X <- matrix(rnorm(m * N, mean, 1), m)
  t(apply(X, 1, function(x) (theta * cumsum(x) - n * theta^2 / 2) / log(10)))
}
L0 <- log10_lr(0, 20000)
cross <- mean(apply(L0, 1, max) >= log10(1 / alpha))
L1 <- log10_lr(theta, 8)
dl <- rbind(data.frame(n = rep(n, each = 40), y = as.vector(L0[1:40, ]), id = rep(1:40, N), k = "data from the null N(0,1)"),
            data.frame(n = rep(n, each = 8), y = as.vector(L1), id = rep(100 + 1:8, N), k = "data from N(0.5,1)"))
dl$k <- factor(dl$k, levels = c("data from the null N(0,1)", "data from N(0.5,1)"))
p2 <- ggplot(dl, aes(n, y, group = id, colour = k)) + geom_line(linewidth = 0.3) +
  geom_hline(yintercept = log10(1 / alpha), colour = PALETTE[["red"]], linetype = "dashed") +
  annotate("text", x = 395, y = log10(1 / alpha) + 0.7, label = "{L[n] == 1/alpha} == 20", parse = TRUE,
           colour = PALETTE[["red"]], hjust = 1, size = 3.4) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "green")])) +
  coord_cartesian(xlim = c(0, N), ylim = c(-8, 8)) +
  labs(x = "n", y = expression(log[10] ~ L[n]), colour = NULL, title = sprintf("Fraction of null paths ever above 20: %.3f", cross)) +
  theme_blog() + theme(legend.position = "bottom")
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.1)), here, "martingale-paths", width = 9.6, height = 3.9)
