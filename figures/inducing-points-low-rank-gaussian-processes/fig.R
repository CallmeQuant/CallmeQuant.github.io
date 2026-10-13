# Inducing-point approximations to Gaussian process regression: SoR, DTC and FITC predictions against
# the exact posterior, and their accuracy as the number of inducing points grows.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(2005)
ELL <- 0.8; SF <- 1; SN <- 0.3
kern <- function(a, b) SF^2 * exp(-0.5 * outer(a, b, "-")^2 / ELL^2)
f_true <- function(x) sin(x) + 0.5 * sin(3 * x)
n <- 300
X <- c(runif(n / 2, 0, 4), runif(n / 2, 6, 10)); y <- f_true(X) + SN * rnorm(n)
xs <- seq(-2.5, 12.5, length.out = 500)
Ky_chol <- chol(kern(X, X) + SN^2 * diag(n))
exact <- function(xq) {
  Ks <- kern(xq, X); a <- backsolve(Ky_chol, forwardsolve(t(Ky_chol), y))
  V <- forwardsolve(t(Ky_chol), t(Ks))
  list(mean = drop(Ks %*% a), var = SF^2 - colSums(V^2))
}
sparse <- function(xq, Z, method) {
  m <- length(Z); Kuu <- kern(Z, Z) + 1e-8 * diag(m); Kuf <- kern(Z, X); Ksu <- kern(xq, Z)
  lam <- if (method == "FITC") SF^2 - colSums(Kuf * solve(Kuu, Kuf)) + SN^2 else rep(SN^2, n)
  A <- Kuu + (Kuf %*% (t(Kuf) / lam))
  mean <- drop(Ksu %*% solve(A, Kuf %*% (y / lam)))
  vw <- rowSums(Ksu * t(solve(A, t(Ksu))))
  var <- if (method == "SoR") vw else SF^2 - rowSums(Ksu * t(solve(Kuu, t(Ksu)))) + vw
  list(mean = mean, var = var)
}
Z <- seq(0.3, 9.7, length.out = 12)
P <- do.call(rbind, lapply(c("exact posterior", "SoR", "DTC", "FITC"), function(meth) {
  r <- if (meth == "exact posterior") exact(xs) else sparse(xs, Z, meth)
  data.frame(x = xs, mean = r$mean, sd = sqrt(pmax(r$var, 0)), method = meth)
}))
P$method <- factor(P$method, levels = c("exact posterior", "SoR", "DTC", "FITC"))
Zd <- data.frame(x = rep(Z, 3), method = factor(rep(c("SoR", "DTC", "FITC"), each = length(Z)), levels = levels(P$method)))
p1 <- ggplot(P, aes(x)) +
  geom_point(data = data.frame(x = X, y = y), aes(x, y), colour = PALETTE[["grey"]], size = 0.35, alpha = 0.6) +
  geom_ribbon(aes(ymin = mean - 1.96 * sd, ymax = mean + 1.96 * sd), fill = PALETTE[["blue"]], alpha = 0.18) +
  geom_line(aes(y = mean), colour = PALETTE[["blue"]], linewidth = 0.7) +
  geom_line(data = data.frame(x = xs, y = f_true(xs)), aes(x, y), colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.35) +
  geom_point(data = Zd, aes(x, y = -2.55), shape = 17, colour = PALETTE[["red"]], size = 1.6) +
  facet_wrap(~method, ncol = 2) + coord_cartesian(xlim = c(-2.5, 12.5), ylim = c(-2.7, 2.7), expand = FALSE) +
  labs(x = "x", y = "f(x)") + theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "sparse-predictions", width = 7.8, height = 5.0)

xt <- seq(-2.5, 12.5, length.out = 400); yt <- f_true(xt) + SN * rnorm(length(xt))
ex <- exact(xt); lpd_ex <- mean(dnorm(yt, ex$mean, sqrt(ex$var + SN^2), log = TRUE))
R <- do.call(rbind, lapply(2:40, function(m) do.call(rbind, lapply(c("SoR", "DTC", "FITC"), function(k) {
  r <- sparse(xt, seq(0.3, 9.7, length.out = m), k)
  data.frame(m = m, method = k, sd_err = sqrt(mean((sqrt(pmax(r$var, 0)) - sqrt(ex$var))^2)),
             lpd = mean(dnorm(yt, r$mean, sqrt(pmax(r$var, 0) + SN^2), log = TRUE)))
}))))
R$method <- factor(R$method, levels = c("SoR", "DTC", "FITC"))
cols <- unname(PALETTE[c("red", "blue", "green")])
p2 <- ggplot(R, aes(m, sd_err, colour = method, linetype = method)) + geom_line(linewidth = 0.7) + scale_y_log10() +
  scale_colour_manual(values = cols) + scale_linetype_manual(values = c("solid", "dashed", "solid")) +
  labs(x = "number of inducing points m", y = "RMS difference of sd", title = "Posterior sd of f vs. exact") + theme_blog()
p3 <- ggplot(R, aes(m, lpd, colour = method, linetype = method)) + geom_line(linewidth = 0.7) +
  geom_hline(yintercept = lpd_ex, linetype = "dotted", colour = PALETTE[["ink"]]) +
  scale_colour_manual(values = cols) + scale_linetype_manual(values = c("solid", "dashed", "solid")) +
  coord_cartesian(ylim = c(-4, 0)) +
  labs(x = "number of inducing points m", y = "mean log predictive density", title = "Held-out predictive density", caption = "dotted: exact GP") + theme_blog()
save_fig(p2 + p3 + plot_layout(guides = "collect") & theme(legend.position = "top"), here, "accuracy-vs-m", width = 7.8, height = 3.4)
