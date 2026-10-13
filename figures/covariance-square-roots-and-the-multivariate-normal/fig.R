# Three square roots of the same covariance matrix applied to the same standard normal draws.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

set.seed(1729)
Sigma <- matrix(c(2.0, 1.2, 1.2, 1.0), 2)
e <- eigen(Sigma, symmetric = TRUE)            # eigenvalues in decreasing order
U <- e$vectors; lam <- e$values
if (U[1, 1] < 0) U[, 1] <- -U[, 1]
if (U[2, 2] < 0) U[, 2] <- -U[, 2]
factors <- list(
  "(a) Z ~ N(0, I[2])" = diag(2),
  "(b) Cholesky: LZ" = t(chol(Sigma)),
  "(c) symmetric root: Sigma^{1/2} * Z" = U %*% diag(sqrt(lam)) %*% t(U),
  "(d) eigen factor: U * Lambda^{1/2} * Z" = U %*% diag(sqrt(lam))
)
labs_parsed <- c("'(a)' ~ Z %~% N(0, I[2])", "'(b) Cholesky:' ~ LZ",
                 "'(c) symmetric root:' ~ Sigma^{1/2} * Z", "'(d) eigen factor:' ~ U * Lambda^{1/2} * Z")
n <- 400
Z <- matrix(rnorm(2 * n), 2)
ang <- atan2(Z[2, ], Z[1, ])
t <- seq(0, 2 * pi, length.out = 300)
circ <- rbind(cos(t), sin(t))
pts <- ell <- arr <- NULL
for (k in seq_along(factors)) {
  A <- factors[[k]]; X <- A %*% Z
  pts <- rbind(pts, data.frame(x = X[1, ], y = X[2, ], ang = ang, panel = labs_parsed[k]))
  for (r in 1:2) { E <- A %*% (r * circ); ell <- rbind(ell, data.frame(x = E[1, ], y = E[2, ], r = r, panel = labs_parsed[k])) }
  arr <- rbind(arr, data.frame(xe = 2 * A[1, ], ye = 2 * A[2, ], col = c("e1", "e2"), panel = labs_parsed[k]))
}
for (d in c("pts", "ell", "arr")) assign(d, transform(get(d), panel = factor(panel, levels = labs_parsed)))
cyc <- hcl(h = seq(15, 375, length.out = 9), c = 55, l = 55)
p <- ggplot() +
  geom_point(data = pts, aes(x, y, colour = ang), size = 0.8, alpha = 0.9) +
  scale_colour_gradientn(colours = cyc, guide = "none") +
  geom_path(data = ell, aes(x, y, group = r), colour = PALETTE[["ink"]], linewidth = 0.35) +
  geom_segment(data = arr, aes(0, 0, xend = xe, yend = ye), colour = ifelse(arr$col == "e1", PALETTE[["red"]], PALETTE[["green"]]),
               linewidth = 0.8, arrow = arrow(length = unit(0.15, "cm"), type = "closed")) +
  facet_wrap(~panel, ncol = 2, labeller = label_parsed) +
  coord_equal(xlim = c(-3.8, 3.8), ylim = c(-3.8, 3.8)) +
  labs(x = expression(x[1]), y = expression(x[2])) +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p, here, "three-square-roots", width = 6.4, height = 6.4)
