# Schur complements in Gaussian conditioning and partial correlation.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(MASS) })

# ---- Figure 1 ------------------------------------------------------------------------------
s1 <- 1; s2 <- 1.5; rho <- 0.7
S <- matrix(c(s1^2, rho * s1 * s2, rho * s1 * s2, s2^2), 2)
Si <- solve(S)
grid <- expand.grid(x2 = seq(-4.5, 4.5, length.out = 250), x1 = seq(-3.2, 3.2, length.out = 250))
grid$q <- Si[1, 1] * grid$x1^2 + 2 * Si[1, 2] * grid$x1 * grid$x2 + Si[2, 2] * grid$x2^2
b <- S[1, 2] / S[2, 2]
cvar <- S[1, 1] - S[1, 2]^2 / S[2, 2]
t <- seq(-3.2, 3.2, length.out = 400)
slices <- do.call(rbind, lapply(c(-2.5, 0, 2.5), function(x2)
  data.frame(x2 = x2, t = t, x = x2 + 1.2 * dnorm(t, b * x2, sqrt(cvar)))))
marg <- data.frame(t = t, x = -4.5 + 1.2 * dnorm(t, 0, s1))
lab_mean <- "E * '[' * X[1] ~ '|' ~ X[2] * '=' * x[2] * ']' == Sigma[12] * Sigma[22]^-1 * x[2]"
lab_cond <- "X[1] ~ '|' ~ X[2] * '=' * x[2] * ',' ~ 'variance' ~ Sigma[11] - Sigma[12] * Sigma[22]^-1 * Sigma[21]"
lab_marg <- "'marginal law of' ~ X[1]"
p1 <- ggplot() +
  geom_vline(xintercept = c(-2.5, 0, 2.5), colour = PALETTE[["light"]], linewidth = 0.3) +
  geom_contour(data = grid, aes(x2, x1, z = q), breaks = c(0.5, 2, 4.5, 8), colour = PALETTE[["grey"]], linewidth = 0.35) +
  geom_abline(aes(intercept = 0, slope = b, colour = "mean", linetype = "mean"), linewidth = 0.6, key_glyph = "path") +
  geom_polygon(data = slices, aes(x, t, group = x2), fill = PALETTE[["blue"]], alpha = 0.18) +
  geom_path(data = slices, aes(x, t, group = x2, colour = "cond", linetype = "cond"), linewidth = 0.7) +
  geom_path(data = marg, aes(x, t, colour = "marg", linetype = "marg"), linewidth = 0.7) +
  scale_colour_manual(values = c(mean = PALETTE[["red"]], cond = PALETTE[["blue"]], marg = PALETTE[["gold"]]),
                      breaks = c("mean", "cond", "marg"),
                      labels = c(parse(text = lab_mean), parse(text = lab_cond), parse(text = lab_marg))) +
  scale_linetype_manual(values = c(mean = "dashed", cond = "solid", marg = "solid"), guide = "none") +
  coord_cartesian(xlim = c(-4.5, 4.5), ylim = c(-3.2, 3.2), expand = FALSE) +
  labs(x = expression(x[2]), y = expression(x[1])) +
  guides(colour = guide_legend(ncol = 2, override.aes = list(linetype = c("dashed", "solid", "solid")))) +
  theme_blog() + theme(legend.text = element_text(size = 8))
save_fig(p1, here, "gaussian-conditioning", width = 6.6, height = 4.2)

# ---- Figure 2 ------------------------------------------------------------------------------
set.seed(7)
n <- 600
Om <- matrix(c(2, 0, -1.2, 0, 2, -1.2, -1.2, -1.2, 2), 3)
X <- mvrnorm(n, rep(0, 3), solve(Om))
d <- data.frame(x1 = X[, 1], x2 = X[, 2], x3 = X[, 3])
d$r1 <- resid(lm(x1 ~ x3 - 1, d)); d$r2 <- resid(lm(x2 ~ x3 - 1, d))
lim <- 1.05 * max(abs(c(d$x1, d$x2)))
qa <- ggplot(d, aes(x1, x2, colour = x3)) + geom_point(size = 0.9, alpha = 0.85) +
  scale_colour_distiller(palette = "RdBu", limits = c(-3, 3), oob = scales::squish, name = expression(x[3])) +
  coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
  labs(x = expression(x[1]), y = expression(x[2]),
       title = bquote("(a)" ~ X[1] ~ "vs" ~ X[2] * ", sample corr." ~ .(sprintf("%.2f", cor(d$x1, d$x2))))) +
  theme_blog() + theme(legend.position = "right", legend.title = element_text())
qb <- ggplot(d, aes(r1, r2)) + geom_point(size = 0.9, alpha = 0.5, colour = PALETTE[["blue"]]) +
  coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
  labs(x = expression("residual of" ~ x[1]), y = expression("residual of" ~ x[2]),
       title = bquote("(b) residuals on" ~ X[3] * ", corr." ~ .(sprintf("%.2f", cor(d$r1, d$r2))))) +
  theme_blog()
save_fig(qa + qb + plot_layout(guides = "collect") & theme(legend.position = "right"), here, "partial-correlation", width = 7.6, height = 3.6)
