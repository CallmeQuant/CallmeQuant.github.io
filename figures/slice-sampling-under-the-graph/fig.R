# Slice sampling with stepping out and shrinkage on a two-component normal mixture.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# Target: 0.55 N(-2.5, 0.5^2) + 0.45 N(2.5, 0.7^2).
f <- function(x) 0.55 * dnorm(x, -2.5, 0.5) + 0.45 * dnorm(x, 2.5, 0.7)

slice_step <- function(x, w) {
  u <- runif(1, 0, f(x))
  L <- x - w * runif(1); R <- L + w
  steps <- list(c(L, R))
  while (f(L) > u) { L <- L - w; steps[[length(steps) + 1]] <- c(L, R) }
  while (f(R) > u) { R <- R + w; steps[[length(steps) + 1]] <- c(L, R) }
  rejected <- numeric(0)
  repeat {
    y <- runif(1, L, R)
    if (f(y) > u) break
    rejected <- c(rejected, y)
    if (y < x) L <- y else R <- y
  }
  list(y = y, u = u, steps = steps, rejected = rejected)
}
run <- function(x0, w, Tn) { out <- numeric(Tn); x <- x0
  for (t in seq_len(Tn)) { x <- slice_step(x, w)$y; out[t] <- x }; out }

# ---- Figure 1: anatomy of one update
xs <- seq(-5, 5, length.out = 1000)
x0 <- -2.05
set.seed(3)
repeat { st <- slice_step(x0, 1); if (st$u > 0.08 && st$u < 0.16 && length(st$rejected) >= 1) break }
u <- st$u
dens <- data.frame(x = xs, f = f(xs))
sl <- data.frame(x = xs, y = ifelse(f(xs) > u, u, NA))
stp <- do.call(rbind, lapply(seq_along(st$steps), function(k)
  data.frame(l = st$steps[[k]][1], r = st$steps[[k]][2], y = -0.035 - 0.022 * (k - 1))))
p1 <- ggplot(dens, aes(x, f)) +
  geom_area(fill = PALETTE[["light"]]) + geom_line(colour = PALETTE[["ink"]], linewidth = 0.6) +
  geom_segment(aes(x = x0, xend = x0, y = 0, yend = f(x0)), colour = PALETTE[["grey"]], linetype = 2) +
  geom_hline(yintercept = u, colour = PALETTE[["grey"]], linewidth = 0.3) +
  geom_line(data = sl, aes(x, y, colour = "slice"), linewidth = 2.2, na.rm = TRUE) +
  geom_errorbarh(data = stp, aes(xmin = l, xmax = r, y = y, colour = "step"), inherit.aes = FALSE,
                 height = 0.012, linewidth = 0.5) +
  geom_point(data = data.frame(x = st$rejected, f = u), aes(colour = "rej"), shape = 4, show.legend = TRUE, size = 3, stroke = 1.3) +
  geom_point(data = data.frame(x = st$y, f = u), aes(colour = "acc"), size = 2.8) +
  annotate("point", x = x0, y = u, colour = PALETTE[["ink"]], size = 1.8) +
  annotate("text", x = x0 + 0.12, y = 0.015, label = "x", parse = TRUE, family = "serif", hjust = 0) +
  annotate("segment", x = -1.1, y = 0.30, xend = x0, yend = u, colour = PALETTE[["grey"]], linewidth = 0.3) +
  annotate("text", x = -1.1, y = 0.315, label = "u %~% U(0, f(x))", parse = TRUE, hjust = 0, family = "serif") +
  scale_colour_manual(values = c(slice = PALETTE[["blue"]], step = PALETTE[["gold"]],
                                 rej = PALETTE[["red"]], acc = PALETTE[["green"]]),
                      breaks = c("slice", "step", "rej", "acc"),
                      labels = c(expression("slice" ~ S[u] == group("{", x * ":" ~ {f(x) > u}, "}")),
                                 expression("stepping out," ~ w == 1), "rejected, interval shrinks",
                                 expression("accepted" ~ x * "'"))) +
  coord_cartesian(xlim = c(-5, 5), ylim = c(-0.035 - 0.022 * nrow(stp), 0.5), expand = FALSE) +
  guides(colour = guide_legend(override.aes = list(linetype = c(1, 1, 0, 0), shape = c(NA, NA, 4, 16),
                                                   size = c(NA, NA, 3, 2.8), linewidth = c(2, 0.6, 0, 0)))) +
  labs(x = "x", y = "f(x)") + theme_blog() +
  theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1), legend.direction = "vertical")
save_fig(p1, here, "one-update", width = 7.4, height = 3.5)

# ---- Figure 2: output for two interval widths
set.seed(42)
Tn <- 20000
ws <- c(0.3, 4); cols <- unname(PALETTE[c("red", "blue")])
panels <- list()
for (i in 1:2) {
  x <- run(-2.5, ws[i], Tn)
  ttl <- sprintf("w = %s: fraction of draws with x > 0 is %.2f (target 0.45)", ws[i], mean(x > 0))
  pt <- ggplot(data.frame(t = seq_len(Tn), x = x), aes(t, x)) + geom_line(colour = cols[i], linewidth = 0.15) +
    coord_cartesian(ylim = c(-5, 5)) + labs(title = ttl, x = if (i == 2) "iteration t" else NULL, y = expression(x[t])) +
    theme_blog() + theme(plot.title = element_text(size = 9.5))
  ph <- ggplot(data.frame(x = x), aes(x)) +
    geom_histogram(aes(y = after_stat(density)), breaks = seq(-5, 5, length.out = 61), fill = cols[i], alpha = 0.45) +
    geom_line(data = dens, aes(x, f), colour = PALETTE[["ink"]], linewidth = 0.5) +
    coord_cartesian(xlim = c(-5, 5), ylim = c(0, 0.62)) +
    labs(x = if (i == 2) "x" else NULL, y = NULL, title = if (i == 1) "histogram and target" else NULL) +
    theme_blog() + theme(plot.title = element_text(size = 9.5))
  panels <- c(panels, list(pt, ph))
}
p2 <- wrap_plots(panels, ncol = 2, widths = c(1.7, 1))
save_fig(p2, here, "interval-width", width = 7.8, height = 4.4)
