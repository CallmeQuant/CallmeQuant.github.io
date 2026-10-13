# Gibbs sampling and random-walk Metropolis for a normal model with semi-conjugate prior.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# Data and prior: y_i ~ N(mu, 1/tau); mu ~ N(0, 10^2); tau ~ Gamma(shape 2, rate 1).
set.seed(2024)
n <- 12
y <- rnorm(n, 1.5, 0.8)
a0 <- 2; b0 <- 1; kappa0 <- 1 / 100

log_post <- function(mu, eta) {
  tau <- exp(eta)
  (n / 2 + a0) * eta - tau * (b0 + sum((y - mu)^2) / 2) - kappa0 * mu^2 / 2
}

gibbs <- function(Tn, mu, tau) {
  out <- matrix(NA, 2 * Tn + 1, 2); out[1, ] <- c(mu, log(tau)); k <- 2
  for (t in seq_len(Tn)) {
    prec <- tau * n + kappa0
    mu <- rnorm(1, tau * sum(y) / prec, 1 / sqrt(prec)); out[k, ] <- c(mu, log(tau)); k <- k + 1
    tau <- rgamma(1, a0 + n / 2, rate = b0 + sum((y - mu)^2) / 2); out[k, ] <- c(mu, log(tau)); k <- k + 1
  }
  out
}

rwm <- function(Tn, x, h) {
  out <- matrix(NA, Tn + 1, 2); out[1, ] <- x; lp <- log_post(x[1], x[2]); acc <- 0
  for (t in seq_len(Tn)) {
    prop <- x + h * rnorm(2)
    lq <- log_post(prop[1], prop[2])
    if (log(runif(1)) < lq - lp) { x <- prop; lp <- lq; acc <- acc + 1 }
    out[t + 1, ] <- x
  }
  list(path = out, acc = acc / Tn)
}

# ---- Figure 1: first moves of each sampler over posterior contours
grid <- expand.grid(mu = seq(-1.2, 3.6, length.out = 250), eta = seq(-2.4, 1.4, length.out = 250))
grid$lp <- mapply(log_post, grid$mu, grid$eta)
grid$p <- exp(grid$lp - max(grid$lp))
start <- c(-0.3, -2.0)
set.seed(1); g <- gibbs(25, start[1], exp(start[2]))
set.seed(2); r <- rwm(50, start, 0.35)$path

panel <- function(path, col, title) {
  d <- data.frame(mu = path[, 1], eta = path[, 2])
  ggplot() +
    geom_contour(data = grid, aes(mu, eta, z = p), breaks = c(0.02, 0.1, 0.3, 0.6, 0.9),
                 colour = PALETTE[["grey"]], linewidth = 0.35) +
    geom_path(data = d, aes(mu, eta), colour = col, linewidth = 0.4) +
    geom_point(data = d, aes(mu, eta), colour = col, size = 0.9) +
    annotate("point", x = start[1], y = start[2], shape = 15, size = 2.2, colour = PALETTE[["ink"]]) +
    coord_cartesian(xlim = c(-1.2, 3.6), ylim = c(-2.4, 1.4), expand = FALSE) +
    labs(title = title, x = expression(mu), y = expression(eta == log ~ tau)) +
    theme_blog()
}
p1 <- panel(g, PALETTE[["blue"]], "Gibbs sampler, 25 sweeps") +
  panel(r, PALETTE[["red"]], "Random-walk Metropolis, 50 steps") + labs(y = NULL)
save_fig(p1, here, "sampler-paths", width = 7.6, height = 3.6)

# ---- Figure 2: traces and autocorrelations of mu for three proposal scales
set.seed(7)
hs <- c(0.05, 0.5, 5)
cols <- unname(PALETTE[c("gold", "blue", "red")])
runs <- lapply(hs, function(h) rwm(4000, c(1.5, 0), h))
traces <- do.call(rbind, lapply(seq_along(hs), function(i) data.frame(
  t = 0:1000, mu = runs[[i]]$path[1:1001, 1],
  lab = sprintf("h = %s, acceptance rate %.2f", hs[i], runs[[i]]$acc))))
traces$lab <- factor(traces$lab, levels = unique(traces$lab))
acfs <- do.call(rbind, lapply(seq_along(hs), function(i) data.frame(
  k = 0:60, r = as.numeric(acf(runs[[i]]$path[1001:4001, 1], lag.max = 60, plot = FALSE)$acf),
  h = factor(paste("h =", hs[i]), levels = paste("h =", hs)))))

pt <- ggplot(traces, aes(t, mu, colour = lab)) + geom_line(linewidth = 0.3) +
  facet_wrap(~lab, ncol = 1) + scale_colour_manual(values = cols, guide = "none") +
  scale_y_continuous(limits = c(0.2, 2.8), breaks = c(0.5, 1.5, 2.5)) +
  labs(x = "iteration t", y = expression(mu[t])) + theme_blog() +
  theme(strip.text = element_text(hjust = 0))
pa <- ggplot(acfs, aes(k, r, colour = h)) + geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_line(linewidth = 0.7) + scale_colour_manual(values = cols) +
  labs(x = "lag k", y = expression("autocorrelation of" ~ mu[t]), title = "Autocorrelation") +
  theme_blog() + theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))
save_fig(pt + pa + plot_layout(widths = c(2.1, 1)), here, "step-size", width = 7.8, height = 4.6)
