# Three copulas with Kendall's tau = 0.5, on the unit square and with fixed margins.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(MASS) })

set.seed(1959)
n <- 1500; tau <- 0.5
rho <- sin(pi * tau / 2); theta <- 2 * tau / (1 - tau)
clayton <- function(n) {               # Marshall-Olkin sampler
  V <- rgamma(n, shape = 1 / theta, rate = 1)
  (1 + matrix(rexp(2 * n), ncol = 2) / V)^(-1 / theta)
}
U_list <- list(
  "Gaussian copula" = pnorm(mvrnorm(n, c(0, 0), matrix(c(1, rho, rho, 1), 2))),
  "Clayton copula" = clayton(n),
  "Survival Clayton copula" = 1 - clayton(n)
)
cols <- unname(PALETTE[c("blue", "red", "green")])

top <- list(); bottom <- list()
for (j in seq_along(U_list)) {
  U <- U_list[[j]]; nm <- names(U_list)[j]
  d <- data.frame(u = U[, 1], v = U[, 2], x = qexp(U[, 1]), y = qnorm(U[, 2]))
  top[[j]] <- ggplot(d, aes(u, v)) + geom_point(colour = cols[j], alpha = 0.6, size = 0.35) +
    annotate("label", x = 0.03, y = 0.97, hjust = 0, vjust = 1, label.size = 0, size = 3,
             label = sprintf("tau = %.2f", cor(d$u, d$v, method = "kendall"))) +
    scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
    scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = nm, x = "u", y = if (j == 1) "v" else NULL) + theme_blog()
  bottom[[j]] <- ggplot(d, aes(x, y)) + geom_point(colour = cols[j], alpha = 0.6, size = 0.35) +
    annotate("label", x = 6.4, y = -3.5, hjust = 1, vjust = 0, label.size = 0, size = 3,
             label = sprintf("Pearson r = %.2f", cor(d$x, d$y))) +
    coord_cartesian(xlim = c(0, 6.5), ylim = c(-3.6, 3.6)) +
    labs(x = "x (Exp(1) margin)", y = if (j == 1) "y (N(0,1) margin)" else NULL) + theme_blog()
}
p <- (top[[1]] | top[[2]] | top[[3]]) / (bottom[[1]] | bottom[[2]] | bottom[[3]])
save_fig(p, here, "three-copulas", width = 8.6, height = 5.8)
