# Union bound for the maximum of m = 20 equicorrelated standard normals (note: Boole's inequality).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

m <- 20; alpha <- 0.05
t <- seq(0, 4.5, length.out = 451)
tail <- pnorm(t, lower.tail = FALSE)

# P(max Z_i > t) for Z_i = sqrt(rho) W + sqrt(1 - rho) E_i, by numerical integration over W
p_max_exceeds <- function(t, rho) {
  if (rho == 1) return(pnorm(t, lower.tail = FALSE))
  if (rho == 0) return(-expm1(m * pnorm(t, log.p = TRUE)))
  sapply(t, function(tt) 1 - integrate(function(w)
    exp(m * pnorm((tt - sqrt(rho) * w) / sqrt(1 - rho), log.p = TRUE)) * dnorm(w),
    -Inf, Inf, rel.tol = 1e-10)$value)
}

union <- m * tail
bonf2 <- m * tail - m * (m - 1) / 2 * tail^2
t_bonf <- qnorm(alpha / m, lower.tail = FALSE)

lab <- c(union = "union bound", exact0 = "exact, rho = 0", exact1 = "exact, rho = 1",
         bonf = "Bonferroni lower bound, rho = 0")
d1 <- rbind(
  data.frame(t = t, p = pmin(union, 1), curve = "union"),
  data.frame(t = t, p = p_max_exceeds(t, 0), curve = "exact0"),
  data.frame(t = t, p = p_max_exceeds(t, 1), curve = "exact1"),
  data.frame(t = t[bonf2 > 0], p = bonf2[bonf2 > 0], curve = "bonf")
)
d1$curve <- factor(d1$curve, levels = names(lab))
p1 <- ggplot(d1, aes(t, p, colour = curve, linetype = curve)) +
  geom_vline(xintercept = t_bonf, colour = PALETTE[["violet"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.7) +
  annotate("text", x = t_bonf + 0.08, y = 0.4, hjust = 0, colour = PALETTE[["violet"]], size = 3.2,
           label = "t == bar(Phi)^{-1}*(alpha/m)", parse = TRUE) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "blue", "gold", "grey")]), labels = lab) +
  scale_linetype_manual(values = c("dashed", "solid", "solid", "dotted"), labels = lab) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^(-4:0)) +
  coord_cartesian(xlim = c(0, 4.5), ylim = c(1e-4, 1.6)) +
  labs(x = "threshold t", y = NULL, title = expression(P(max[i] ~ Z[i] > t) * ", " ~ m == 20)) +
  theme_blog() +
  guides(colour = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)) +
  theme(legend.position = "bottom", legend.key.width = unit(1.4, "lines"),
        legend.text = element_text(size = rel(0.78)))

rhos <- c(0, 0.5, 0.9, 1)
d2 <- do.call(rbind, lapply(rhos, function(r)
  data.frame(t = t, ratio = union / p_max_exceeds(t, r), rho = sprintf("rho == %g", r))))
d2$rho <- factor(d2$rho, levels = sprintf("rho == %g", rhos))
p2 <- ggplot(d2, aes(t, ratio, colour = rho)) +
  geom_hline(yintercept = 1, colour = PALETTE[["grey"]], linetype = "dashed", linewidth = 0.4) +
  annotate("segment", x = t_bonf, xend = t_bonf, y = 0.8, yend = 28, colour = PALETTE[["violet"]],
           linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = CYCLE[1:4], labels = function(x) parse(text = x)) +
  scale_y_log10(breaks = c(1, 2, 5, 10, 20)) +
  coord_cartesian(xlim = c(0, 4.5), ylim = c(0.8, 70)) +
  labs(x = "threshold t", y = NULL, title = "union bound / exact probability") +
  theme_blog() +
  theme(legend.position = "bottom")

save_fig(p1 + p2, here, "union-bound-maximum", width = 8.0, height = 3.9)
