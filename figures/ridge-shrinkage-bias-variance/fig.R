# Ridge shrinkage: ridge path geometry for two correlated covariates; shrinkage factors and MSE.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# ---- Figure 1 ------------------------------------------------------------------------------
rho <- 0.9
G <- matrix(c(1, rho, rho, 1), 2)
b_ols <- c(2.0, -0.3)
ridge <- function(l) drop(solve(G + l * diag(2), G %*% b_ols))
lams <- c(0, 10^seq(-3, 2.5, length.out = 400))
path <- t(sapply(lams, ridge))
grid <- expand.grid(b1 = seq(-1.2, 3, length.out = 250), b2 = seq(-1.2, 2.2, length.out = 250))
D <- cbind(grid$b1 - b_ols[1], grid$b2 - b_ols[2])
grid$rss <- rowSums((D %*% G) * D)
circ <- do.call(rbind, lapply(c(0.5, 1, 1.5), function(r) {
  t <- seq(0, 2 * pi, length.out = 200); data.frame(x = r * cos(t), y = r * sin(t), r = r) }))
marks <- data.frame(l = c(0.1, 1, 5)); marks <- cbind(marks, t(sapply(marks$l, ridge)))
names(marks) <- c("l", "b1", "b2")
vecs <- data.frame(x = 0, y = 0, xe = 0.9 * c(1, 1) / sqrt(2), ye = 0.9 * c(1, -1) / sqrt(2), lab = c("v[1]", "v[2]"))

pa <- ggplot() +
  geom_path(data = circ, aes(x, y, group = r), colour = PALETTE[["light"]], linewidth = 0.4) +
  geom_contour(data = grid, aes(b1, b2, z = rss), breaks = c(0.05, 0.2, 0.5, 1, 1.6, 2.4),
               colour = PALETTE[["grey"]], linewidth = 0.35) +
  geom_segment(data = vecs, aes(x, y, xend = xe, yend = ye), colour = PALETTE[["green"]],
               arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.5) +
  geom_text(data = vecs, aes(xe + 0.08, ye + ifelse(ye > 0, 0.08, -0.12), label = lab), parse = TRUE,
            colour = PALETTE[["green"]], family = "serif") +
  geom_path(data = data.frame(b1 = path[, 1], b2 = path[, 2]), aes(b1, b2), colour = PALETTE[["blue"]], linewidth = 0.9) +
  geom_point(data = marks, aes(b1, b2), colour = PALETTE[["blue"]], size = 1.6) +
  geom_label(data = marks, aes(b1 + 0.05, b2 - 0.15, label = paste0("lambda==", l)), parse = TRUE, hjust = 0,
             size = 3, colour = PALETTE[["blue"]], label.size = 0, label.padding = unit(0.05, "lines"), family = "serif") +
  annotate("point", x = b_ols[1], y = b_ols[2], colour = PALETTE[["red"]], size = 2.6) +
  annotate("text", x = b_ols[1] + 0.12, y = b_ols[2], label = "hat(beta)^OLS", parse = TRUE, hjust = 0,
           colour = PALETTE[["red"]], family = "serif") +
  annotate("point", x = 0, y = 0, shape = 3, size = 3) +
  coord_equal(xlim = c(-1.2, 3), ylim = c(-1.2, 2.2), expand = FALSE) +
  labs(x = expression(beta[1]), y = expression(beta[2]), title = "(a) Ridge path in the parameter plane") +
  theme_blog()

lp <- 10^seq(-3, 2.5, length.out = 400)
pp <- t(sapply(lp, ridge))
dpp <- rbind(data.frame(l = lp, b = pp[, 1], k = "beta1"), data.frame(l = lp, b = pp[, 2], k = "beta2"))
pb <- ggplot(dpp, aes(l, b, colour = k)) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = CYCLE[1:2],
                      labels = c(expression(hat(beta)[list(lambda, 1)]), expression(hat(beta)[list(lambda, 2)]))) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  labs(x = expression(lambda ~ "(log scale)"), y = NULL, title = "(b) Coefficients along the path") +
  theme_blog() + theme(legend.position = c(0.8, 0.85))
save_fig(pa + pb + plot_layout(widths = c(1, 1.15)), here, "ridge-geometry", width = 7.6, height = 3.5)

# ---- Figure 2 ------------------------------------------------------------------------------
d2 <- c(50, 20, 8, 3, 1, 0.3)
alpha <- c(1.0, -0.8, 0.6, 0.5, -0.4, 0.3)
sigma2 <- 1
lam <- 10^seq(-3, 3, length.out = 400)
S <- outer(lam, d2, function(l, d) d / (d + l))
bias2 <- rowSums(sweep((1 - S)^2, 2, alpha^2, "*"))
vari <- rowSums(sweep(S^2, 2, sigma2 / d2, "*"))
mse <- bias2 + vari
mse_ols <- sum(sigma2 / d2)
lam_hk <- 2 * sigma2 / max(alpha^2)

ds <- data.frame(l = rep(lam, length(d2)), s = as.vector(S),
                 j = factor(rep(paste0("d[", seq_along(d2), "]^2==", d2), each = length(lam)),
                            levels = paste0("d[", seq_along(d2), "]^2==", d2)))
qa <- ggplot(ds, aes(l, s, colour = j)) + geom_line(linewidth = 0.8) +
  scale_colour_manual(values = CYCLE, labels = scales::label_parse()) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(0, 1.02)) +
  labs(x = expression(lambda ~ "(log scale)"), y = expression(s[j](lambda) == d[j]^2 / (d[j]^2 + lambda)),
       title = "(a) Shrinkage factors") +
  theme_blog() + theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0), legend.text = element_text(size = 8), legend.key.height = unit(0.75, "lines"), legend.background = element_rect(fill = "white", colour = NA))

dm <- rbind(data.frame(l = lam, v = mse, k = "MSE"), data.frame(l = lam, v = bias2, k = "squared bias"),
            data.frame(l = lam, v = vari, k = "variance"))
k <- which.min(mse)
qb <- ggplot(dm, aes(l, v, colour = k, linetype = k)) +
  annotate("rect", xmin = min(lam), xmax = lam_hk, ymin = -Inf, ymax = Inf, fill = PALETTE[["light"]], alpha = 0.7) +
  geom_hline(yintercept = mse_ols, colour = PALETTE[["grey"]], linetype = "dotted") +
  annotate("text", x = max(lam), y = mse_ols + 0.12, label = "OLS", hjust = 1, vjust = 0, colour = PALETTE[["grey"]], size = 3.2) +
  geom_line(linewidth = 0.8) +
  annotate("point", x = lam[k], y = mse[k], colour = PALETTE[["ink"]], size = 1.6) +
  scale_colour_manual(values = c(MSE = PALETTE[["ink"]], `squared bias` = PALETTE[["red"]], variance = PALETTE[["blue"]])) +
  scale_linetype_manual(values = c(MSE = "solid", `squared bias` = "dashed", variance = "dashed")) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(0, 5.6)) +
  labs(x = expression(lambda ~ "(log scale)"), y = NULL, title = expression("(b) Mean squared error of" ~ hat(beta)[lambda])) +
  theme_blog() + theme(legend.position = c(0.98, 0.84), legend.justification = c(1, 1))
save_fig(qa + qb, here, "shrinkage-mse", width = 7.6, height = 3.4)
