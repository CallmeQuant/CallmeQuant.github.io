# Random Fourier features: Monte Carlo approximations of the Gaussian and Laplace kernels, and the error as
# a function of the number of features.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(2007)
ELL <- 1
delta <- seq(-6, 6, length.out = 601)
kg <- function(d) exp(-0.5 * d^2 / ELL^2); kl <- function(d) exp(-abs(d) / ELL)
rff <- function(d, om, b) rowMeans(2 * cos(outer(d, om) + rep(b, each = length(d))) * rep(cos(b), each = length(d)))
mk <- function(kf, sampler, title) {
  out <- data.frame(d = delta, v = kf(delta), what = "exact")
  for (D in c(20, 500)) { om <- sampler(D); b <- runif(D, 0, 2 * pi)
    out <- rbind(out, data.frame(d = delta, v = rff(delta, om, b), what = sprintf("D = %d", D))) }
  out$what <- factor(out$what, levels = c("exact", "D = 20", "D = 500")); out$panel <- title; out
}
d1 <- rbind(mk(kg, function(D) rnorm(D, 0, 1 / ELL), "Gaussian~kernel*','~omega %~% N(0, l^{-2})"),
            mk(kl, function(D) rcauchy(D, 0, 1 / ELL), "Laplace~kernel*','~omega %~% Cauchy(0, l^{-1})"))
d1$panel <- factor(d1$panel, levels = unique(d1$panel))
p1 <- ggplot(d1, aes(d, v, colour = what, linewidth = what)) + geom_line() + facet_wrap(~panel, labeller = label_parsed) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "gold", "blue")])) + scale_linewidth_manual(values = c(1, 0.5, 0.5)) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-0.65, 1.3), expand = FALSE) +
  labs(x = expression(delta == x - x*"'"), y = expression(hat(k)[D](delta))) + theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "kernel-approximations", width = 7.8, height = 3.4)

Ds <- unique(as.integer(exp(seq(log(4), log(4096), length.out = 12))) %/% 2 * 2)
reps <- 400; d0 <- 1; k0 <- kg(d0); k2 <- kg(2 * d0); grid <- seq(-4, 4, length.out = 161)
res <- do.call(rbind, lapply(Ds, function(D) {
  s <- replicate(reps, {
    om <- rnorm(D, 0, 1 / ELL); b <- runif(D, 0, 2 * pi); om2 <- rnorm(D / 2, 0, 1 / ELL)
    c(rff(d0, om, b) - k0, max(abs(rff(grid, om, b) - kg(grid))),
      mean(cos(om2 * d0)) - k0, max(abs(rowMeans(cos(outer(grid, om2))) - kg(grid))))
  })
  data.frame(D = D, sd_ph = sd(s[1, ]), sup_ph = mean(s[2, ]), sd_pr = sd(s[3, ]), sup_pr = mean(s[4, ]))
}))
d2 <- rbind(data.frame(D = res$D, v = res$sd_ph, f = sqrt((1 + 0.5 * k2 - k0^2) / res$D), what = "random phase"),
            data.frame(D = res$D, v = res$sd_pr, f = sqrt((1 + k2 - 2 * k0^2) / res$D), what = "(cos, sin) pairs"))
p2 <- ggplot(d2, aes(D, colour = what)) + geom_line(aes(y = f), linewidth = 0.6) + geom_point(aes(y = v, shape = what), size = 1.8) +
  scale_x_log10() + scale_y_log10() + scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  scale_shape_manual(values = c(15, 16)) +
  labs(x = "number of features D", y = "error", title = expression("sd of" ~ hat(k)[D](delta) ~ "at" ~ delta == 1), caption = "points: simulated; lines: formula") + theme_blog()
d3 <- rbind(data.frame(D = res$D, v = res$sup_ph, what = "random phase"), data.frame(D = res$D, v = res$sup_pr, what = "(cos, sin) pairs"))
p3 <- ggplot(d3, aes(D, v, colour = what, shape = what)) + geom_line(linewidth = 0.5) + geom_point(size = 1.8) +
  geom_line(data = data.frame(D = res$D, v = 2.5 / sqrt(res$D)), aes(D, v), inherit.aes = FALSE, linetype = "dashed", colour = PALETTE[["grey"]]) +
  scale_x_log10() + scale_y_log10() + scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) + scale_shape_manual(values = c(15, 16)) +
  labs(x = "number of features D", y = NULL, title = expression("mean of" ~ sup ~ "|" * hat(k)[D] - k * "|" ~ "over" ~ "|" * delta * "|" <= 4), caption = "dashed: slope -1/2") + theme_blog()
save_fig(p2 + p3 + plot_layout(guides = "collect") & theme(legend.position = "top"), here, "error-vs-features", width = 7.8, height = 3.6)
