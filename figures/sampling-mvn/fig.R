# Cholesky factors of the covariance and of the precision matrix of a stationary AR(1) vector,
# and draws obtained from the precision factor by back-substitution (note: sampling the multivariate normal).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

phi <- 0.9
ar1_cov <- function(d) phi^abs(outer(1:d, 1:d, "-")) / (1 - phi^2)
ar1_prec <- function(d) {
  Q <- diag(c(1, rep(1 + phi^2, d - 2), 1))
  Q[cbind(1:(d - 1), 2:d)] <- -phi
  Q[cbind(2:d, 1:(d - 1))] <- -phi
  Q
}

d_small <- 30
L_S <- t(chol(ar1_cov(d_small)))   # chol() returns the upper factor
L_Q <- t(chol(ar1_prec(d_small)))

heat <- function(M, title) {
  df <- expand.grid(i = 1:nrow(M), j = 1:ncol(M))
  df$v <- abs(M[cbind(df$i, df$j)])
  df <- df[df$v > 1e-12, ]
  ggplot(df, aes(j, i, fill = v)) +
    geom_tile() +
    annotate("text", x = d_small, y = 1, hjust = 1, vjust = 1, size = 3,
             label = paste(nrow(df), "nonzeros")) +
    scale_fill_gradientn(colours = c("#ffffff", PALETTE[["blue"]], PALETTE[["ink"]]),
                         limits = c(0, max(abs(M))), guide = "none") +
    scale_y_reverse(breaks = c(1, d_small), expand = c(0, 0)) +
    scale_x_continuous(breaks = c(1, d_small), expand = c(0, 0)) +
    coord_fixed() +
    labs(x = NULL, y = NULL, title = title) +
    theme_blog() +
    theme(panel.grid.major = element_blank(), axis.line = element_blank(),
          panel.border = element_rect(colour = PALETTE[["grey"]], fill = NA, linewidth = 0.3))
}
pa <- heat(L_S, expression("(a)" ~ L[Sigma]))
pb <- heat(L_Q, expression("(b)" ~ L[Q] * "," ~ Q == Sigma^{-1}))

# (c) draws y = L_Q^{-T} z by back-substitution with the bidiagonal factor
d <- 300
U <- chol(ar1_prec(d))          # upper factor, equal to L_Q^T
set.seed(7)
paths <- do.call(rbind, lapply(1:3, function(j) {
  z <- rnorm(d)
  data.frame(i = 1:d, y = backsolve(U, z), draw = factor(j))
}))
sdv <- 1 / sqrt(1 - phi^2)
pc <- ggplot(paths, aes(i, y, colour = draw)) +
  annotate("rect", xmin = 1, xmax = d, ymin = -2 * sdv, ymax = 2 * sdv, fill = PALETTE[["light"]]) +
  geom_line(linewidth = 0.35) +
  scale_colour_manual(values = CYCLE[1:3], guide = "none") +
  coord_cartesian(xlim = c(1, d), ylim = c(-8.5, 8.5), expand = FALSE) +
  labs(x = "coordinate i", y = NULL, title = expression("(c) draws" ~ y == L[Q]^{-T} * z * "," ~ d == 300)) +
  theme_blog()

save_fig(pa + pb + pc + plot_layout(widths = c(1, 1, 1.9)), here, "precision-cholesky", width = 8.6, height = 3.1)
