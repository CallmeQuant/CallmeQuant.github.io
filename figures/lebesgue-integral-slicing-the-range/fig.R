# Riemann sums versus Lebesgue sums, the distribution function, and dyadic simple functions.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

f <- function(x) 1 + 0.5 * sin(2 * pi * x) + 0.3 * x * cos(7 * pi * x)
x <- seq(0, 1, length.out = 20001); fx <- f(x)
I <- sum((fx[-1] + fx[-length(fx)]) / 2) * diff(x)[1]
G <- function(t) mean(fx >= t)
curve_df <- data.frame(x = x[seq(1, 20001, 5)], y = fx[seq(1, 20001, 5)])

n <- 10; e <- seq(0, 1, length.out = n + 1)
rect_r <- data.frame(xmin = e[-(n + 1)], xmax = e[-1], ymin = 0, ymax = f(e[-(n + 1)]))
R <- sum(rect_r$ymax * diff(e))
p1 <- ggplot() +
  geom_rect(data = rect_r, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = PALETTE[["blue"]], alpha = 0.25, colour = PALETTE[["blue"]], linewidth = 0.25) +
  geom_line(data = curve_df, aes(x, y), colour = PALETTE[["ink"]], linewidth = 0.7) +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c("0", "0.25", "0.5", "0.75", "1")) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.75), expand = FALSE) +
  labs(x = "x", y = "f(x)", title = sprintf("Riemann: slice the domain, sum = %.3f", R)) +
  theme_blog() + theme(plot.title = element_text(size = 9.5))

h <- 0.25; levels <- seq(h, max(fx) + h, by = h)
bands <- do.call(rbind, lapply(seq_along(levels), function(k) {
  y <- levels[k]; m <- as.integer(fx >= y)
  if (!any(m == 1)) return(NULL)
  d <- diff(c(0, m, 0)); s <- which(d == 1); en <- which(d == -1) - 1
  data.frame(xmin = x[s], xmax = x[en], ymin = y - h, ymax = y, k = factor(k))
}))
L <- sum(sapply(levels, function(y) h * G(y)))
cols <- rep(CYCLE[1:5], 3)
p2 <- ggplot() +
  geom_rect(data = bands, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = k), alpha = 0.3) +
  geom_segment(data = bands, aes(x = xmin, xend = xmax, y = ymax, yend = ymax, colour = k), linewidth = 0.9) +
  geom_line(data = curve_df, aes(x, y), colour = PALETTE[["ink"]], linewidth = 0.7) +
  scale_fill_manual(values = cols, guide = "none") + scale_colour_manual(values = cols, guide = "none") +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c("0", "0.25", "0.5", "0.75", "1")) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.75), expand = FALSE) +
  labs(x = "x", y = NULL, title = sprintf("Lebesgue: slice the range, sum = %.3f", L)) +
  theme_blog() + theme(plot.title = element_text(size = 9.5))

tt <- seq(0, 1.75, length.out = 600)
dG <- data.frame(t = tt, g = sapply(tt, G))
lv <- data.frame(y = levels, g = sapply(levels, G), k = factor(seq_along(levels)))
lv <- lv[lv$g > 0, ]
p3 <- ggplot() +
  geom_ribbon(data = dG, aes(y = t, xmin = 0, xmax = g), fill = PALETTE[["grey"]], alpha = 0.3, orientation = "y") +
  geom_path(data = dG, aes(g, t), colour = PALETTE[["ink"]], linewidth = 0.7) +
  geom_segment(data = lv, aes(x = 0, xend = g, y = y, yend = y, colour = k), linewidth = 0.9) +
  scale_colour_manual(values = cols, guide = "none") +
  coord_cartesian(xlim = c(0, 1.05), ylim = c(0, 1.75), expand = FALSE) +
  labs(x = "measure of {f >= t}", y = "t", title = sprintf("Area = integral of f = %.3f", I)) +
  theme_blog() + theme(plot.title = element_text(size = 9.5))
save_fig(p1 + p2 + p3 + plot_layout(widths = c(1, 1, 0.9)), here, "riemann-vs-lebesgue", width = 10.4, height = 3.4)

ps <- lapply(1:4, function(nn) {
  phi <- pmin(floor(fx * 2^nn) / 2^nn, nn)
  d <- data.frame(x = x, phi = phi)[seq(1, 20001, 5), ]
  hl <- (1:(nn * 2^nn)) / 2^nn; hl <- hl[hl <= 1.75]
  ggplot(d) +
    geom_hline(yintercept = hl, colour = PALETTE[["grey"]], linewidth = 0.15, alpha = 0.6) +
    geom_area(aes(x, phi), fill = PALETTE[["blue"]], alpha = 0.25) +
    geom_line(aes(x, phi), colour = PALETTE[["blue"]], linewidth = 0.4) +
    geom_line(data = curve_df, aes(x, y), colour = PALETTE[["ink"]], linewidth = 0.6) +
    scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c("0", "0.25", "0.5", "0.75", "1")) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.75), expand = FALSE) +
    labs(x = "x", y = if (nn == 1) "phi_n(x)" else NULL, title = sprintf("n = %d: integral %.3f", nn, mean(phi))) +
    theme_blog() + theme(panel.grid.major.y = element_blank())
})
save_fig(wrap_plots(ps, nrow = 1), here, "dyadic-approximations", width = 10.6, height = 2.9)
