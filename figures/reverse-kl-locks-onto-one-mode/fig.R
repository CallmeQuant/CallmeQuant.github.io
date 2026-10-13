# Gaussian fits to a two-component mixture under reverse and forward KL, and mean-field fits to a
# correlated Gaussian (note: zero-forcing in reverse Kullback-Leibler fits).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

zg <- seq(-10, 10, length.out = 2001); wz <- dnorm(zg); wz <- wz / sum(wz)   # quadrature for N(0, 1)
log_p <- function(x, a, w) {
  u <- log(w) - 0.5 * (x - a)^2; v <- log1p(-w) - 0.5 * (x + a)^2
  pmax(u, v) + log1p(exp(-abs(u - v))) - 0.5 * log(2 * pi)
}
reverse_kl <- function(m, s, a, w) { x <- m + s * zg; sum(wz * (dnorm(x, m, s, log = TRUE) - log_p(x, a, w))) }
reverse_profile <- function(m, a, w) {
  ls <- seq(log(0.2), log(8), length.out = 120)
  v <- sapply(ls, function(l) reverse_kl(m, exp(l), a, w)); k <- which.min(v)
  r <- optimize(function(l) reverse_kl(m, exp(l), a, w), c(ls[max(k - 1, 1)], ls[min(k + 1, length(ls))]))
  c(kl = r$objective, s = exp(r$minimum))
}
forward_profile <- function(m, a, w) {
  xx <- seq(-a - 12, a + 12, length.out = 20001); lp <- log_p(xx, a, w)
  H <- -sum(exp(lp) * lp) * diff(xx[1:2])
  mu <- (2 * w - 1) * a; v <- 1 + 4 * w * (1 - w) * a^2
  -H + 0.5 * log(2 * pi * exp(1) * (v + (m - mu)^2))
}

# Figure 1
a <- 3.2; w <- 0.6
mg <- seq(-6, 6, length.out = 241)
rev <- t(sapply(mg, reverse_profile, a = a, w = w))
fwd <- sapply(mg, forward_profile, a = a, w = w)
idx <- which(diff(sign(diff(rev[, "kl"]))) > 0) + 1
fits <- t(sapply(idx, function(i) {
  r <- optimize(function(m) reverse_profile(m, a, w)[["kl"]], mg[c(i - 1, i + 1)])
  c(m = r$minimum, s = reverse_profile(r$minimum, a, w)[["s"]], kl = r$objective)
}))
fits <- fits[order(fits[, "kl"]), , drop = FALSE]
mu_p <- (2 * w - 1) * a; sd_p <- sqrt(1 + 4 * w * (1 - w) * a^2)
x <- seq(-7.5, 7.5, length.out = 800)
labs1 <- c("target p", "argmin KL(q || p)", "its other local minimum", "argmin KL(p || q)")
d1 <- rbind(
  data.frame(x = x, y = exp(log_p(x, a, w)), what = labs1[1]),
  data.frame(x = x, y = dnorm(x, fits[1, "m"], fits[1, "s"]), what = labs1[2]),
  data.frame(x = x, y = dnorm(x, fits[2, "m"], fits[2, "s"]), what = labs1[3]),
  data.frame(x = x, y = dnorm(x, mu_p, sd_p), what = labs1[4]))
d1$what <- factor(d1$what, levels = labs1)
p1 <- ggplot(d1, aes(x, y, colour = what, linetype = what)) +
  geom_area(data = subset(d1, what == labs1[1]), fill = PALETTE[["light"]], colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red", "red", "blue")])) +
  scale_linetype_manual(values = c("solid", "solid", "dashed", "solid")) +
  coord_cartesian(xlim = c(-7.5, 7.5), ylim = c(0, 0.52), expand = FALSE) +
  labs(x = "x", y = "density", title = "Target and Gaussian fits") +
  guides(colour = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)) +
  theme_blog() + theme(legend.text = element_text(size = 8))
d2 <- rbind(data.frame(m = mg, v = rev[, "kl"], what = "reverse: min_s KL(N(m, s^2) || p)"),
            data.frame(m = mg, v = fwd, what = "forward: min_s KL(p || N(m, s^2))"))
d2$what <- factor(d2$what, levels = unique(d2$what))
p2 <- ggplot(d2, aes(m, v, colour = what)) +
  geom_hline(yintercept = c(-log(w), -log(1 - w)), colour = PALETTE[["grey"]], linetype = "dotted") +
  annotate("text", x = -5.9, y = c(-log(w), -log(1 - w)) - 0.05, label = c("-log(w)", "-log(1-w)"),
           hjust = 0, vjust = 1, size = 3, colour = PALETTE[["grey"]], family = "serif") +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(0, 3), expand = FALSE) +
  labs(x = "location m of q = N(m, s^2)", y = "divergence (profiled over s)", title = "Profiles in the location") +
  guides(colour = guide_legend(ncol = 1)) +
  theme_blog() + theme(legend.text = element_text(size = 8))
save_fig(p1 + p2, here, "mixture-fits", width = 8.2, height = 3.8)

# Figure 2
mg2 <- seq(-5, 5, length.out = 201)
d3 <- do.call(rbind, lapply(c(1.5, 2.5, 2.8, 3.5), function(aa)
  data.frame(m = mg2, v = sapply(mg2, function(m) reverse_profile(m, aa, 0.5)[["kl"]]), a = paste0("a = ", aa))))
p3 <- ggplot(d3, aes(m, v, colour = a)) +
  geom_hline(yintercept = log(2), colour = PALETTE[["grey"]], linetype = "dotted") +
  annotate("text", x = 4.95, y = log(2) + 0.04, label = "log 2", hjust = 1, vjust = 0, size = 3,
           colour = PALETTE[["grey"]], family = "serif") +
  geom_line(linewidth = 0.8) + scale_colour_blog() +
  coord_cartesian(xlim = c(-5, 5), ylim = c(0, 1.8), expand = FALSE) +
  labs(x = "location m", y = "min_s KL(N(m, s^2) || p_a)") + theme_blog()
save_fig(p3, here, "separation-profiles", width = 6.4, height = 3.4)

# Figure 3
ell <- function(S, r = 2) { t <- seq(0, 2 * pi, length.out = 300); L <- t(chol(S)); e <- L %*% rbind(cos(t), sin(t)) * r
  data.frame(x1 = e[1, ], x2 = e[2, ]) }
labs3 <- c("target N(0, Sigma)", "reverse KL: v_i = 1/Lambda_ii", "forward KL: v_i = Sigma_ii")
d4 <- do.call(rbind, lapply(c(0.6, 0.95), function(rho) {
  S <- matrix(c(1, rho, rho, 1), 2); Lam <- solve(S)
  rbind(cbind(ell(S), what = labs3[1]), cbind(ell(diag(1 / diag(Lam))), what = labs3[2]),
        cbind(ell(diag(diag(S))), what = labs3[3])) |> transform(rho = paste0("rho = ", rho))
}))
d4$what <- factor(d4$what, levels = labs3)
p4 <- ggplot(d4, aes(x1, x2, colour = what, linetype = what)) + geom_path(linewidth = 0.8) +
  facet_wrap(~rho) + coord_equal(xlim = c(-2.6, 2.6), ylim = c(-2.6, 2.6)) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red", "blue")])) +
  scale_linetype_manual(values = c("solid", "solid", "dashed")) +
  labs(x = "x1", y = "x2") + theme_blog() + guides(colour = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)) +
  theme(legend.position = "bottom", strip.text = element_text(hjust = 0))
save_fig(p4, here, "mean-field-ellipses", width = 7.0, height = 3.9)
