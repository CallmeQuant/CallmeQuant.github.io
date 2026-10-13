# Optimal transport: (a) optimal assignment between two 7-point clouds (squared Euclidean cost,
# all permutations enumerated); (b) the monotone (quantile) coupling between two densities on
# the line; entropic transport plans (log-domain Sinkhorn) for three regularization levels.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(dplyr) })

set.seed(1781)
n <- 7
X <- cbind(rnorm(n) * 0.55 - 1.3, rnorm(n) * 0.55)
ang <- runif(n, 0, 2 * pi)
Y <- cbind(1.1 * cos(ang) + 1.3, 1.1 * sin(ang)) + matrix(rnorm(2 * n) * 0.08, ncol = 2)
C <- outer(1:n, 1:n, Vectorize(function(i, j) sum((X[i, ] - Y[j, ])^2)))
perms <- function(v) if (length(v) == 1) matrix(v, 1) else
  do.call(rbind, lapply(seq_along(v), function(i) cbind(v[i], perms(v[-i]))))
P <- perms(1:n)
cost <- apply(P, 1, function(p) sum(C[cbind(1:n, p)]))
best <- P[which.min(cost), ]

seg <- data.frame(x = X[, 1], y = X[, 2], xend = Y[best, 1], yend = Y[best, 2])
pts <- rbind(data.frame(x = X[, 1], y = X[, 2], set = "x_i (mass 1/7 each)"),
             data.frame(x = Y[, 1], y = Y[, 2], set = "y_j (mass 1/7 each)"))
LABS <- c(expression(x[i] ~ "(mass 1/7 each)"), expression(y[j] ~ "(mass 1/7 each)"))
pa <- ggplot() +
  geom_segment(data = seg, aes(x, y, xend = xend, yend = yend), colour = PALETTE[["grey"]], linewidth = 0.45) +
  geom_point(data = pts, aes(x, y, colour = set, shape = set), size = 2.3) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), labels = LABS) +
  scale_shape_manual(values = c(16, 15), labels = LABS) +
  coord_equal(xlim = c(-2.6, 2.8), ylim = c(-1.7, 2.3)) +
  labs(x = NULL, y = NULL, title = "(a) optimal assignment, squared Euclidean cost") +
  theme_blog() + theme(legend.position = c(0.01, 0.99), legend.justification = c(0, 1))

f_mu <- function(x) 0.55 * dnorm(x, -1.6, 0.55) + 0.45 * dnorm(x, 1.0, 0.7)
f_nu <- function(x) dnorm(x, 0.6, 1.0)
grid <- seq(-5, 5, length.out = 4001)
Fc <- cumsum(f_mu(grid)); Fc <- Fc / Fc[length(Fc)]
Gc <- cumsum(f_nu(grid)); Gc <- Gc / Gc[length(Gc)]
Finv <- function(t) approx(Fc, grid, t, ties = "ordered", rule = 2)$y
Ginv <- function(t) approx(Gc, grid, t, ties = "ordered", rule = 2)$y

xs <- seq(-4, 4, length.out = 600); top <- 1; s <- 0.9
tq <- (0:14 + 0.5) / 15
arr <- data.frame(x = Finv(tq), y = top - 0.02, xend = Ginv(tq), yend = 0.02)
dens <- rbind(data.frame(x = xs, lo = top, hi = top + s * f_mu(xs), which = "mu"),
              data.frame(x = xs, lo = -s * f_nu(xs), hi = 0, which = "nu"))
pb <- ggplot(dens) +
  geom_ribbon(aes(x, ymin = lo, ymax = hi, fill = which), alpha = 0.25) +
  geom_line(data = filter(dens, which == "mu"), aes(x, hi, colour = which), linewidth = 0.7) +
  geom_line(data = filter(dens, which == "nu"), aes(x, lo, colour = which), linewidth = 0.7) +
  geom_hline(yintercept = c(0, top), linewidth = 0.3, colour = PALETTE[["ink"]]) +
  geom_segment(data = arr, aes(x, y, xend = xend, yend = yend), linewidth = 0.3,
               arrow = arrow(length = unit(0.12, "cm"), type = "closed"), colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), labels = scales::label_parse()) +
  scale_fill_manual(values = unname(PALETTE[c("blue", "red")]), guide = "none") +
  coord_cartesian(xlim = c(-4, 4), ylim = c(-0.5, 1.5)) +
  labs(x = "x", y = NULL, title = expression("(b) quantile coupling " * F^{-1} * (t) %->% G^{-1} * (t))) +
  theme_blog() +
  theme(axis.text.y = element_blank(), axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid.major = element_blank(), legend.position = c(1, 1), legend.justification = c(1, 1))
save_fig(pa + pb + plot_layout(widths = c(1, 1.15)), here, "couplings", width = 8.4, height = 3.5)

# ---------------------------------------------------------------- entropic plans
m <- 120
z <- seq(-4, 4, length.out = m)
a <- f_mu(z); a <- a / sum(a)
b <- f_nu(z); b <- b / sum(b)
Cz <- outer(z, z, function(x, y) (x - y)^2)
lse_rows <- function(M) { mx <- apply(M, 1, max); mx + log(rowSums(exp(M - mx))) }
sinkhorn_log <- function(a, b, C, eps, iters) {
  f <- numeric(length(a)); g <- numeric(length(b))
  for (it in seq_len(iters)) {
    f <- eps * log(a) - eps * lse_rows(sweep(-C, 2, g, `+`) / eps)
    g <- eps * log(b) - eps * lse_rows(t(sweep(-C, 1, f, `+`)) / eps)
  }
  exp((outer(f, g, `+`) - C) / eps)
}
Tmap <- data.frame(x = z, y = Ginv(approx(grid, Fc, z)$y))
plans <- bind_rows(lapply(c(1, 0.1, 0.01), function(eps) {
  Pm <- sinkhorn_log(a, b, Cz, eps, 3000)
  expand.grid(x = z, y = z) |> mutate(p = as.vector(Pm) / max(Pm), eps = paste("epsilon ==", eps))
}))
plans$eps <- factor(plans$eps, levels = unique(plans$eps))
pe <- ggplot(plans, aes(x, y, fill = p)) + geom_raster() +
  geom_line(data = Tmap, aes(x, y), inherit.aes = FALSE, colour = PALETTE[["red"]], linetype = "dashed",
            linewidth = 0.4) +
  scale_fill_gradientn(colours = c("white", PALETTE[["blue"]], PALETTE[["ink"]]), guide = "none") +
  facet_wrap(~eps, nrow = 1, labeller = label_parsed) +
  coord_equal(xlim = c(-4, 4), ylim = c(-4, 4), expand = FALSE) +
  labs(x = expression(x ~ "(source " * mu * ")"), y = expression(y ~ "(target " * nu * ")")) +
  theme_blog() + theme(panel.grid.major = element_blank(), strip.text = element_text(hjust = 0),
                       panel.spacing = unit(1.6, "lines"))
save_fig(pe, here, "entropic", width = 8.4, height = 3.2)
