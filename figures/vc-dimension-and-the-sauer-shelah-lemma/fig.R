# VC dimension. Figure 1: three points shattered by half-planes; Radon partitions of four points.
# Figure 2: growth functions against 2^n and the Sauer-Shelah bound.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
POS <- PALETTE[["blue"]]; NEG <- PALETTE[["red"]]

P3 <- rbind(c(0, 0), c(1, 0.15), c(0.35, 1))
sep_line <- function(P, y) {
  if (all(y) || all(!y)) return(NULL)
  odd <- if (sum(y) == 1) y else !y
  s <- P[odd, ]; others <- P[!odd, , drop = FALSE]
  m <- 0.5 * (s + colMeans(others)); d <- others[2, ] - others[1, ]
  w <- c(-d[2], d[1]); if (sum(w * (s - m)) < 0) w <- -w
  if (sum(y) != 1) w <- -w
  list(w = w, b = -sum(w * m))
}
labs8 <- as.matrix(expand.grid(c3 = 0:1, c2 = 0:1, c1 = 0:1)[, 3:1]) == 1
pts <- do.call(rbind, lapply(1:8, function(k) data.frame(x = P3[, 1], y = P3[, 2], lab = labs8[k, ], panel = k)))
shade <- do.call(rbind, lapply(1:8, function(k) {
  y <- labs8[k, ]; sl <- sep_line(P3, y)
  if (is.null(sl)) {
    if (all(y)) return(data.frame(x = c(-0.35, 1.35, 1.35, -0.35), y = c(-0.3, -0.3, 1.3, 1.3), panel = k))
    return(NULL)
  }
  xs <- c(-0.6, 1.6); ys <- -(sl$w[1] * xs + sl$b) / sl$w[2]
  top <- if (sl$w[2] > 0) 10 else -10
  data.frame(x = c(xs, rev(xs)), y = c(ys, top, top), panel = k)
}))
lines8 <- do.call(rbind, lapply(1:8, function(k) {
  sl <- sep_line(P3, labs8[k, ]); if (is.null(sl)) return(NULL)
  xs <- c(-0.6, 1.6); data.frame(x = xs, y = -(sl$w[1] * xs + sl$b) / sl$w[2], panel = k)
}))
blank <- theme_void() + theme(panel.border = element_rect(colour = PALETTE[["light"]], fill = NA),
                              strip.text = element_blank(), plot.title = element_text(size = 10, colour = PALETTE[["ink"]]))
top <- ggplot() +
  geom_polygon(data = shade, aes(x, y, group = panel), fill = POS, alpha = 0.1) +
  geom_line(data = lines8, aes(x, y, group = panel), colour = PALETTE[["ink"]], linewidth = 0.4) +
  geom_point(data = pts, aes(x, y, fill = lab), shape = 21, colour = ifelse(pts$lab, POS, NEG), size = 2, stroke = 0.9) +
  scale_fill_manual(values = c(`TRUE` = POS, `FALSE` = "white"), guide = "none") +
  facet_wrap(~panel, nrow = 2) + coord_cartesian(xlim = c(-0.35, 1.35), ylim = c(-0.3, 1.3), expand = FALSE) +
  labs(title = "Three points: all 8 labelings are realized by half-planes (shaded side = label 1)") + blank

radon <- function(P, I, ttl) {
  J <- setdiff(1:4, I); df <- data.frame(x = P[, 1], y = P[, 2], lab = 1:4 %in% I)
  g <- ggplot()
  for (S in list(list(I, POS), list(J, NEG))) {
    Q <- P[S[[1]], , drop = FALSE]
    if (nrow(Q) == 2) g <- g + geom_path(data = data.frame(x = Q[, 1], y = Q[, 2]), aes(x, y), colour = S[[2]], linewidth = 0.7)
    if (nrow(Q) >= 3) g <- g + geom_polygon(data = data.frame(x = Q[, 1], y = Q[, 2]), aes(x, y), fill = S[[2]], alpha = 0.12, colour = S[[2]])
  }
  g + geom_point(data = df, aes(x, y, fill = lab), shape = 21, colour = ifelse(df$lab, POS, NEG), size = 2.4, stroke = 1) +
    scale_fill_manual(values = c(`TRUE` = POS, `FALSE` = "white"), guide = "none") +
    coord_cartesian(xlim = c(-0.3, 1.6), ylim = c(-0.25, 1.3), expand = FALSE) + labs(title = ttl) + blank
}
b1 <- radon(rbind(c(0, 0), c(1.2, 0.1), c(1.1, 1), c(0.1, 0.9)), c(1, 3), "Four points, convex position: the diagonals cross")
b2 <- radon(rbind(c(0, 0), c(1.3, 0.1), c(0.6, 1.1), c(0.62, 0.42)), 4, "One point inside the others' triangle")
save_fig(top / (b1 | b2) + plot_layout(heights = c(1, 1.1)), here, "shattering", width = 7.8, height = 5)

n <- 1:30
gf <- rbind(data.frame(n = n, v = n + 1, cls = "thresholds (d = 1): n + 1", d = 1),
            data.frame(n = n, v = choose(n, 2) + n + 1, cls = "intervals (d = 2): choose(n,2) + n + 1", d = 2),
            data.frame(n = n, v = n^2 - n + 2, cls = "half-planes in R^2 (d = 3): n^2 - n + 2", d = 3))
gf$sauer <- mapply(function(nn, d) sum(choose(nn, 0:d)), gf$n, gf$d)
gf$cls <- factor(gf$cls, levels = unique(gf$cls))
g2 <- ggplot(gf, aes(n, v, colour = cls)) +
  geom_line(data = data.frame(n = n, v = 2^n), aes(n, v), inherit.aes = FALSE, linetype = "dashed", colour = PALETTE[["ink"]]) +
  geom_line(aes(y = sauer), linewidth = 0.4) + geom_point(size = 1.3) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "green", "red")])) +
  coord_cartesian(ylim = c(1, 1e5)) + guides(colour = guide_legend(ncol = 1)) +
  labs(x = "n", y = "growth function", caption = "dashed: 2^n;  thin lines: Sauer-Shelah bound") +
  theme_blog()
save_fig(g2, here, "growth-functions", width = 6.4, height = 4)
