# Coordinate ascent variational inference with a mean-field Gaussian family for a bivariate
# Gaussian target (note: variational inference and the evidence lower bound).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

mu <- c(1, 0.5)
m0 <- c(-1.5, 2.5)   # initial means; variances start at their optimal values 1 - rho^2

cavi <- function(rho, sweeps) {
  Lam <- solve(matrix(c(1, rho, rho, 1), 2))
  m <- m0; path <- matrix(m, nrow = 1)
  for (s in seq_len(sweeps)) {
    m[1] <- mu[1] - Lam[1, 2] / Lam[1, 1] * (m[2] - mu[2]); path <- rbind(path, m)
    m[2] <- mu[2] - Lam[2, 1] / Lam[2, 2] * (m[1] - mu[1]); path <- rbind(path, m)
  }
  path
}
kl_mf <- function(m, rho) {
  Lam <- solve(matrix(c(1, rho, rho, 1), 2)); d <- m - mu
  0.5 * drop(t(d) %*% Lam %*% d) - 0.5 * log(1 - rho^2)
}
ellipse <- function(S, r, n = 200) {
  a <- seq(0, 2 * pi, length.out = n)
  E <- t(mu + r * t(chol(S)) %*% rbind(cos(a), sin(a)))
  data.frame(z1 = E[, 1], z2 = E[, 2])
}

rho <- 0.8
S <- matrix(c(1, rho, rho, 1), 2)
ell <- rbind(transform(ellipse(S, 1), what = "target p", r = 1), transform(ellipse(S, 2), what = "target p", r = 2),
             transform(ellipse(diag(1 - rho^2, 2), 1), what = "optimal q*", r = 1),
             transform(ellipse(diag(1 - rho^2, 2), 2), what = "optimal q*", r = 2))
ell$grp <- paste(ell$what, ell$r)
path <- as.data.frame(cavi(rho, 8)); names(path) <- c("z1", "z2"); path$what <- "CAVI means"
cols <- c("target p" = PALETTE[["ink"]], "optimal q*" = PALETTE[["red"]], "CAVI means" = PALETTE[["blue"]])
pa <- ggplot() +
  geom_path(data = ell, aes(z1, z2, group = grp, colour = what, linetype = what), linewidth = 0.6) +
  geom_path(data = path, aes(z1, z2, colour = what, linetype = what), linewidth = 0.5) +
  geom_point(data = path, aes(z1, z2, colour = what), size = 0.9) +
  annotate("text", x = m0[1] + 0.2, y = m0[2] + 0.2, label = "start", hjust = 0, size = 3.2, colour = PALETTE[["blue"]]) +
  scale_colour_manual(values = cols, breaks = names(cols)) +
  scale_linetype_manual(values = c("target p" = "solid", "optimal q*" = "dashed", "CAVI means" = "solid"), breaks = names(cols)) +
  coord_fixed(xlim = c(-2.4, 3.6), ylim = c(-2.2, 3.2)) +
  labs(x = expression(z[1]), y = expression(z[2]), title = expression("(a)" ~ rho == 0.8 * ": contours at radii 1 and 2")) +
  guides(colour = guide_legend(nrow = 2, override.aes = list(shape = c(NA, NA, 16))),
         linetype = guide_legend(nrow = 2)) +
  theme_blog() +
  theme(legend.position = "bottom")

rhos <- c(0.5, 0.8, 0.95)
kld <- do.call(rbind, lapply(rhos, function(r) {
  P <- cavi(r, 30)[-1, ]
  data.frame(sweep = seq_len(nrow(P)) / 2, kl = apply(P, 1, kl_mf, rho = r), rho = sprintf("rho == %g", r))
}))
floors <- data.frame(rho = sprintf("rho == %g", rhos), floor = -0.5 * log(1 - rhos^2))
pb <- ggplot(kld, aes(sweep, kl, colour = rho)) +
  geom_hline(data = floors, aes(yintercept = floor, colour = rho), linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = CYCLE[1:3], labels = function(x) parse(text = x)) +
  scale_y_log10(breaks = c(0.1, 0.2, 0.5, 1, 2, 5)) +
  coord_cartesian(xlim = c(0, 15), ylim = c(0.1, 6)) +
  labs(x = "sweep", y = expression(KL(q[t] ~ "||" ~ p) == log ~ p(x) - ELBO(q[t])),
       title = "(b) gap after each coordinate update") +
  theme_blog() +
  theme(legend.position = "bottom")

save_fig(pa + pb + plot_layout(widths = c(1, 1.15)), here, "cavi-gaussian", width = 8.0, height = 4.1)
