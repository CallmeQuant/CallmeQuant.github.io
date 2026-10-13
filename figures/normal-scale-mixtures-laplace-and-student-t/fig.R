# Normal scale mixtures: Laplace and Student t with unit variance, mixing laws, EM weights.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1974)
nu <- 5; b <- 1 / sqrt(2); s_t <- sqrt((nu - 2) / nu); M <- 2e6
Z <- rnorm(M)
X_lap <- sqrt(rexp(M, rate = 1 / (2 * b^2))) * Z
X_t <- sqrt((nu - 2) / rchisq(M, nu)) * Z
edges <- seq(-8, 8, length.out = 65); mids <- (edges[-1] + edges[-65]) / 2; w <- edges[2] - edges[1]
hist_df <- function(X, lab) {
  h <- hist(X[abs(X) < 8], breaks = edges, plot = FALSE)$counts
  data.frame(x = mids, d = h / (M * w), k = lab)[h > 20, ]
}
x <- seq(-8, 8, length.out = 600)
dens <- rbind(data.frame(x = x, d = dnorm(x), k = "normal"),
              data.frame(x = x, d = exp(-abs(x) / b) / (2 * b), k = "Laplace"),
              data.frame(x = x, d = dt(x / s_t, nu) / s_t, k = "Student t5"))
dens$k <- factor(dens$k, levels = c("normal", "Laplace", "Student t5"))
pts <- rbind(hist_df(X_lap, "Laplace"), hist_df(X_t, "Student t5"))
pts$k <- factor(pts$k, levels = levels(dens$k))
cols <- unname(PALETTE[c("grey", "blue", "red")])
p1 <- ggplot() + geom_line(data = dens, aes(x, d, colour = k), linewidth = 0.8) +
  geom_point(data = pts, aes(x, d, colour = k), shape = 1, size = 0.9, show.legend = FALSE) +
  scale_colour_manual(values = cols, labels = c("normal", "Laplace", expression("Student" ~ t[5]))) +
  scale_y_log10(breaks = 10^(-6:0), labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(1e-6, 1.2), xlim = c(-8, 8)) +
  labs(x = "x", y = NULL, title = "Densities, unit variance (log scale)") +
  theme_blog() + theme(legend.position = c(1, 1), legend.justification = c(1, 1))

v <- seq(1e-3, 4, length.out = 500)
mix <- rbind(data.frame(v = v, d = dexp(v, rate = 1 / (2 * b^2)), k = "Laplace: Exp, mean 1"),
             data.frame(v = v, d = dchisq((nu - 2) / v, nu) * (nu - 2) / v^2, k = "t5: 3/G, G ~ chi2(5)"))
p2 <- ggplot(mix, aes(v, d, colour = k)) + geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 1, colour = PALETTE[["grey"]], linewidth = 0.9) +
  annotate("text", x = 1.08, y = 1.5, label = "normal: V = 1", hjust = 0, size = 3, colour = PALETTE[["grey"]]) +
  scale_colour_manual(values = cols[2:3], labels = c(expression("Laplace:" ~ V %~% "Exp, mean 1"), expression(t[5] * ":" ~ V == 3/G * "," ~ G %~% chi[5]^2))) + coord_cartesian(xlim = c(0, 4), ylim = c(0, 1.6)) +
  labs(x = "v", y = NULL, title = "Mixing densities of V") +
  theme_blog() + theme(legend.position = c(1, 0.85), legend.justification = c(1, 1), legend.text = element_text(size = 8))

r <- seq(-6, 6, length.out = 801)
wts <- rbind(data.frame(r = r, w = 1, k = "normal"),
             data.frame(r = r, w = (nu + 1) / (nu + r^2), k = "t5"),
             data.frame(r = r, w = 4 / (3 + r^2), k = "t3"),
             data.frame(r = r, w = 2 / (1 + r^2), k = "Cauchy"),
             data.frame(r = r, w = ifelse(abs(r) > 0.15, 1 / abs(r), NA), k = "Laplace"))
wts$k <- factor(wts$k, levels = c("normal", "t5", "t3", "Cauchy", "Laplace"))
p3 <- ggplot(wts, aes(r, w, colour = k)) + geom_line(linewidth = 0.8, na.rm = TRUE) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "red", "gold", "green", "blue")]), labels = c("normal", expression(t[5]), expression(t[3]), "Cauchy", "Laplace")) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(0, 2.6)) +
  labs(x = "standardized residual r", y = NULL, title = expression("EM weights" ~ E * "[" * V^{-1} ~ "|" ~ r * "]")) +
  theme_blog() + theme(legend.position = c(0, 1), legend.justification = c(0, 1), legend.key.size = unit(0.7, "lines"))
save_fig(p1 + p2 + p3 + plot_layout(widths = c(1.2, 1, 1)), here, "scale-mixtures", width = 10, height = 3.3)
