# Maximum likelihood under misspecification.
# Figure 1: normal linear model fitted to a nonlinear regression; the projection depends on the design.
# Figure 2: exponential model fitted to log-normal data; coverage of model-based and sandwich intervals.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
set.seed(1982)

# ---- Figure 1 ----
m <- function(z) 2 * sin(z)
n <- 300; zz <- seq(-3.2, 3.2, length.out = 400)
panel <- function(mu, sd, ttl) {
  z <- rnorm(n, mu, sd); y <- m(z) + rnorm(n, 0, 0.5)
  b <- coef(lm(y ~ z))
  slope <- 2 * cos(mu) * exp(-sd^2 / 2)
  inter <- 2 * sin(mu) * exp(-sd^2 / 2) - slope * mu
  dens <- dnorm(zz, mu, sd)
  lines_df <- rbind(
    data.frame(z = zz, y = m(zz), what = "truth"),
    data.frame(z = zz, y = inter + slope * zz, what = "projection"),
    data.frame(z = zz, y = b[1] + b[2] * zz, what = "fit"))
  lines_df$what <- factor(lines_df$what, levels = c("truth", "projection", "fit"))
  ggplot() +
    geom_ribbon(data = data.frame(z = zz, lo = -4.6, hi = -4.6 + 1.3 * dens / max(dens)),
                aes(z, ymin = lo, ymax = hi), fill = PALETTE[["gold"]], alpha = 0.35) +
    geom_point(data = data.frame(z = z, y = y), aes(z, y), colour = PALETTE[["grey"]], alpha = 0.55, size = 0.7) +
    geom_line(data = lines_df, aes(z, y, colour = what, linetype = what), linewidth = 0.7) +
    scale_colour_manual(values = unname(PALETTE[c("ink", "blue", "red")]),
                        labels = c(expression(E(Y ~ "|" ~ Z == z) == 2 * sin ~ z), expression("projection" ~ beta[0]^"*" + beta[1]^"*" * z), expression("fit," ~ n == 300))) +
    scale_linetype_manual(values = c("solid", "solid", "dashed"),
                          labels = c(expression(E(Y ~ "|" ~ Z == z) == 2 * sin ~ z), expression("projection" ~ beta[0]^"*" + beta[1]^"*" * z), expression("fit," ~ n == 300))) +
    coord_cartesian(xlim = c(-3.2, 3.2), ylim = c(-4.6, 4.6), expand = FALSE) +
    labs(x = expression(z), y = expression(y), title = parse(text = sprintf("%s * ',' ~ beta[1]^'*' == '%.2f'", ttl, slope))) +
    theme_blog() + theme(plot.title = element_text(size = 9.5))
}
p1 <- panel(0, 1, "'Design A:' ~ Z %~% N(0, 1)")
p2 <- panel(1.5, 0.5, "'Design B:' ~ Z %~% N(1.5, 0.5^2)") + labs(y = NULL)
save_fig(p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = "top"),
         here, "design-dependence", width = 7.8, height = 3.8)

# ---- Figure 2 ----
ss <- seq(0.2, 1.6, by = 0.1); z975 <- qnorm(0.975); R <- 20000
sfine <- seq(0.15, 1.65, length.out = 300)
asym <- rbind(data.frame(s = sfine, cov = 2 * pnorm(z975 / sqrt(expm1(sfine^2))) - 1, kind = "model-based"),
              data.frame(s = sfine, cov = 0.95, kind = "sandwich"))
sims <- do.call(rbind, lapply(c(100, 1000), function(nn) do.call(rbind, lapply(ss, function(s) {
  lam_star <- exp(-s^2 / 2)
  x <- matrix(rlnorm(R * nn, 0, s), R, nn)
  xbar <- rowMeans(x); lam <- 1 / xbar
  v <- rowSums((x - xbar)^2) / (nn - 1)
  data.frame(s = s, n = paste("n =", nn),
             cov = c(mean(abs(lam - lam_star) <= z975 * lam / sqrt(nn)),
                     mean(abs(lam - lam_star) <= z975 * sqrt(v / nn) / xbar^2)),
             kind = c("model-based", "sandwich"))
}))))
p3 <- ggplot() +
  geom_vline(xintercept = sqrt(log(2)), colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  annotate("text", x = sqrt(log(2)) + 0.02, y = 0.76, label = "CV = 1", hjust = 0, colour = PALETTE[["grey"]], size = 3.2) +
  geom_line(data = asym, aes(s, cov, colour = kind), linewidth = 0.8) +
  geom_point(data = sims, aes(s, cov, colour = kind, shape = n), fill = "white", size = 1.8, stroke = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) +
  scale_shape_manual(values = c(21, 22)) +
  coord_cartesian(xlim = c(0.15, 1.65), ylim = c(0.38, 1.02)) +
  labs(x = expression("log-normal shape" ~ s), y = "coverage of nominal 95% interval",
       caption = expression("lines:" ~ n %->% infinity * "; points: simulation")) +
  theme_blog()
save_fig(p3, here, "coverage", width = 6.6, height = 3.8)
