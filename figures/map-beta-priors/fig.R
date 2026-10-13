# Symmetric Beta priors for a coin's head probability and the posteriors after five heads in five
# tosses, with posterior modes (note: maximum a posteriori estimation).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

t <- seq(0.0005, 0.9995, length.out = 800)
alphas <- c(1, 2, 5, 10); n1 <- 5; n2 <- 0

pri <- do.call(rbind, lapply(alphas, function(a)
  data.frame(theta = t, density = dbeta(t, a, a), shape = sprintf("alpha == beta ~ \"=\" ~ %g", a))))
pri$shape <- factor(pri$shape, levels = unique(pri$shape))
post <- do.call(rbind, lapply(alphas, function(a)
  data.frame(theta = t, density = dbeta(t, n1 + a, n2 + a), shape = sprintf("Beta(%g, %g)", n1 + a, n2 + a))))
post$shape <- factor(post$shape, levels = unique(post$shape))
modes <- do.call(rbind, lapply(alphas, function(a) {
  m <- (n1 + a - 1) / (n1 + n2 + 2 * a - 2)
  data.frame(theta = m, density = dbeta(min(m, 0.9995), n1 + a, n2 + a), shape = sprintf("Beta(%g, %g)", n1 + a, n2 + a))
}))
modes$shape <- factor(modes$shape, levels = levels(post$shape))

base <- function(p, title) {
  p +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 6.6)) +
    labs(x = expression(theta), y = "density", title = title) +
    theme_blog() +
    theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1),
          legend.background = element_rect(fill = "white", colour = NA))
}
p1 <- base(ggplot(pri, aes(theta, density, colour = shape)) + geom_line(linewidth = 0.7) +
             scale_colour_blog(labels = function(x) parse(text = x)),
           expression("Priors" ~ "Beta" * (list(alpha, alpha))))
p2 <- base(ggplot(post, aes(theta, density, colour = shape)) +
             geom_vline(xintercept = 1, colour = PALETTE[["grey"]], linetype = "dashed", linewidth = 0.4) +
             annotate("text", x = 0.985, y = 6.25, label = "MLE", hjust = 1, size = 3.2, colour = PALETTE[["grey"]]) +
             geom_line(linewidth = 0.7) + geom_point(data = modes, size = 1.8) + scale_colour_blog(),
           "Posteriors, 5 heads in 5 tosses (dots: MAP)") +
  theme(axis.title.y = element_blank())
save_fig(p1 + p2, here, "beta-densities", width = 7.6, height = 3.3)
