# Escaping mass and domination.
# (a) f_n = n 1_(0,1/n]; (b) f_n = 1_[n,n+1); (c) f_n(x) = (1 + x/n)^(-n), dominated by (1 + x/2)^(-2).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

ns <- c(1, 2, 4, 8)
cols <- setNames(unname(PALETTE[c("gold", "green", "red", "blue")]), paste0("n = ", ns))
da <- do.call(rbind, lapply(ns, function(n)
  data.frame(x = c(0, 0, 1 / n, 1 / n, 1.2), y = c(0, n, n, 0, 0), n = paste0("n = ", n))))
db <- do.call(rbind, lapply(ns, function(n)
  data.frame(x = c(0, n, n, n + 1, n + 1, 10), y = c(0, 0, 1, 1, 0, 0), n = paste0("n = ", n))))
x <- seq(0, 6, length.out = 400)
dc <- do.call(rbind, lapply(ns[-1], function(n) data.frame(x = x, y = (1 + x / n)^(-n), n = paste0("n = ", n))))
lvl <- names(cols)
da$n <- factor(da$n, lvl); db$n <- factor(db$n, lvl); dc$n <- factor(dc$n, lvl)

pa <- ggplot(da, aes(x, y, colour = n)) + geom_path(linewidth = 0.7) +
  scale_colour_manual(values = cols) + coord_cartesian(ylim = c(0, 9)) +
  labs(x = "x", y = NULL, title = expression("(a) " * f[n] == n %.% 1[group("(", list(0, 1/n), "]")])) +
  theme_blog() + theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))
pb <- ggplot(db, aes(x, y, colour = n)) + geom_path(linewidth = 0.7) +
  scale_colour_manual(values = cols, guide = "none") + coord_cartesian(ylim = c(0, 1.4)) +
  labs(x = "x", y = NULL, title = expression("(b) " * f[n] == 1[group("[", list(n, n + 1), ")")])) +
  theme_blog()
pc <- ggplot() +
  geom_area(data = data.frame(x = x, y = (1 + x / 2)^(-2)), aes(x, y), fill = PALETTE[["light"]]) +
  geom_line(data = dc, aes(x, y, colour = n), linewidth = 0.7) +
  geom_line(data = data.frame(x = x, y = exp(-x)), aes(x, y), linetype = "dashed", colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = cols, guide = "none") +
  labs(x = "x", y = NULL, title = expression("(c) " * {f[n] == (1 + x/n)^{-n}} * ", " * {n >= 2})) +
  theme_blog()
save_fig(pa + pb + pc, here, "escaping-mass", width = 9.4, height = 3.2)
