# Upper-tail bounds for S - np, S ~ Binomial(n, p) with n = 200: exact tail versus Chebyshev,
# Hoeffding, Bernstein and the optimized Chernoff bound, for p = 0.5 and p = 0.05.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

n <- 200
kl <- function(q, p) q * log(q / p) + (1 - q) * log((1 - q) / (1 - p))
lv <- c("exact", "Chebyshev", "Hoeffding", "Bernstein", "Chernoff (optimized)")
d <- do.call(rbind, lapply(list(c(0.5, 45), c(0.05, 30)), function(pt) {
  p <- pt[1]; t <- seq(0.5, pt[2], length.out = 300)
  v <- n * p * (1 - p); M <- max(p, 1 - p)
  panel <- sprintf("p = %g, np(1-p) = %g", p, v)
  rbind(
    data.frame(t = t, b = pbinom(ceiling(n * p + t) - 1, n, p, lower.tail = FALSE), what = "exact", panel = panel),
    data.frame(t = t, b = pmin(1, v / t^2), what = "Chebyshev", panel = panel),
    data.frame(t = t, b = exp(-2 * t^2 / n), what = "Hoeffding", panel = panel),
    data.frame(t = t, b = exp(-t^2 / (2 * v + 2 * M * t / 3)), what = "Bernstein", panel = panel),
    data.frame(t = t, b = exp(-n * kl(p + t / n, p)), what = "Chernoff (optimized)", panel = panel))
}))
d$what <- factor(d$what, levels = lv)
d$panel <- factor(d$panel, levels = unique(d$panel))
p <- ggplot(d, aes(t, b, colour = what, linetype = what)) +
  geom_step(data = subset(d, what == "exact"), linewidth = 0.7) +
  geom_line(data = subset(d, what != "exact"), linewidth = 0.7) +
  scale_y_log10(limits = c(1e-12, 1.5), labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "gold", "red", "green", "blue")]), drop = FALSE) +
  scale_linetype_manual(values = c("solid", "dotted", "dashed", "dotdash", "solid"), drop = FALSE) +
  facet_wrap(~panel, scales = "free_x") +
  labs(x = "deviation t", y = "P(S - np >= t) and bounds") +
  theme_blog() + theme(legend.position = "bottom")
save_fig(p, here, "binomial-tails", width = 7.8, height = 3.8)
