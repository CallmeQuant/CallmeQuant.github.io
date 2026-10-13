# Huber location M-estimators: score functions and asymptotic variance versus threshold k.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(20261012)
EPS <- 0.1; TAU <- 5
pdf_of <- list(
  normal = function(x) dnorm(x),
  contam = function(x) (1 - EPS) * dnorm(x) + EPS * dnorm(x, sd = TAU),
  t3 = function(x) dt(x, 3)
)
cdf_of <- list(
  normal = function(x) pnorm(x),
  contam = function(x) (1 - EPS) * pnorm(x) + EPS * pnorm(x, sd = TAU),
  t3 = function(x) pt(x, 3)
)
draw <- list(
  normal = function(m) rnorm(m),
  contam = function(m) ifelse(runif(m) < EPS, TAU, 1) * rnorm(m),
  t3 = function(m) rt(m, 3)
)
labs_law <- c(normal = "N(0,1)", contam = "0.9 N(0,1) + 0.1 N(0,25)", t3 = "Student t[3]")
cols <- unname(PALETTE[c("blue", "red", "green")])

avar <- function(name, k) {
  B <- 2 * integrate(function(x) x^2 * pdf_of[[name]](x), 0, k)$value + 2 * k^2 * (1 - cdf_of[[name]](k))
  V <- 2 * cdf_of[[name]](k) - 1
  B / V^2
}
huber_location <- function(X, k, iters = 60) {
  lo <- apply(X, 1, min); hi <- apply(X, 1, max)
  for (i in seq_len(iters)) {
    mid <- (lo + hi) / 2
    s <- rowSums(pmin(pmax(X - mid, -k), k))
    lo <- ifelse(s > 0, mid, lo); hi <- ifelse(s > 0, hi, mid)
  }
  (lo + hi) / 2
}

ks <- seq(0.05, 4, length.out = 160)
k_sim <- c(0.25, 0.75, 1.345, 2, 3); n <- 200; R <- 4000
curves <- do.call(rbind, lapply(names(pdf_of), function(nm)
  data.frame(k = ks, v = sapply(ks, function(k) avar(nm, k)), law = labs_law[[nm]])))
sims <- do.call(rbind, lapply(names(pdf_of), function(nm)
  data.frame(k = k_sim, v = sapply(k_sim, function(k) {
    X <- matrix(draw[[nm]](R * n), R, n); n * var(huber_location(X, k)) }), law = labs_law[[nm]])))
lev <- unname(labs_law)
curves$law <- factor(curves$law, levels = lev); sims$law <- factor(sims$law, levels = lev)

u <- seq(-3, 3, length.out = 601)
score <- rbind(
  data.frame(u = u, phi = u, g = "mean: phi(u) == u", part = 1),
  data.frame(u = u, phi = pmin(pmax(u, -1.345), 1.345), g = "Huber, k == 1.345", part = 1),
  data.frame(u = u[u < 0], phi = -1, g = "median: sign(u)", part = 1),
  data.frame(u = u[u > 0], phi = 1, g = "median: sign(u)", part = 2)
)
score$g <- factor(score$g, levels = c("mean: phi(u) == u", "Huber, k == 1.345", "median: sign(u)"))
p1 <- ggplot(score, aes(u, phi, colour = g, group = interaction(g, part))) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "blue", "gold")]),
                      labels = c(expression("mean:" ~ varphi(u) == u), expression("Huber," ~ k == 1.345),
                                 expression("median:" ~ sign(u)))) +
  coord_cartesian(xlim = c(-3, 3), ylim = c(-3, 3)) +
  labs(x = expression("residual" ~ u == x - theta), y = expression(varphi(u)), title = "Score functions") +
  guides(colour = guide_legend(ncol = 1)) + theme_blog() + theme(legend.text = element_text(size = 8))

p2 <- ggplot(curves, aes(k, v, colour = law)) +
  geom_vline(xintercept = 1.345, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  geom_point(data = sims, shape = 21, fill = "white", size = 1.9, stroke = 0.9) +
  scale_colour_manual(values = cols, labels = c(expression(N(0, 1)), expression(0.9 ~ N(0, 1) + 0.1 ~ N(0, 25)),
                                                expression("Student" ~ t[3]))) +
  coord_cartesian(xlim = c(0, 4.05), ylim = c(0.9, 3.5)) +
  labs(x = expression("threshold" ~ k), y = expression(sigma^2 * (k) == B / V^2),
       title = expression("Asymptotic variance of" ~ sqrt(n) * (hat(theta)[n] - theta[0]))) +
  guides(colour = guide_legend(ncol = 1)) + theme_blog() + theme(legend.text = element_text(size = 8))

save_fig(p1 + p2 + plot_layout(widths = c(1, 1.35)), here, "huber-variance", width = 7.6, height = 4.3)
