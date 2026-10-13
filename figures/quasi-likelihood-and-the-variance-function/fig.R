# Quasi-likelihood with a misspecified variance function: gamma-type data with Var(Y) = phi mu^2 and
# log mu = b0 + b1 x, fitted by quasi-score equations with working variance V(mu) = mu^q.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })
set.seed(1974)
BETA <- c(0.5, 1); PHI <- 0.5; n <- 200
x <- seq(0, 3, length.out = n); X <- cbind(1, x); mu0 <- as.vector(exp(X %*% BETA))

asym_cov <- function(q, mu = mu0) {
  D <- X * mu; A <- D / mu^q
  bread <- solve(t(A) %*% D)
  bread %*% (t(A * (PHI * mu^2)) %*% A) %*% t(bread)
}
fit <- function(y, q) {
  b <- c(log(mean(y)), 0)
  for (it in 1:50) {
    mu <- as.vector(exp(X %*% b)); D <- X * mu; A <- D / mu^q
    step <- solve(t(A) %*% D, t(A) %*% (y - mu)); b <- b + as.vector(step)
    if (max(abs(step)) < 1e-10) break
  }
  mu <- as.vector(exp(X %*% b)); D <- X * mu; A <- D / mu^q
  bread <- solve(t(A) %*% D)
  phi_hat <- sum((y - mu)^2 / mu^q) / (n - 2)
  list(b = b, se_m = sqrt(phi_hat * bread[2, 2]),
       se_s = sqrt((bread %*% (t(A * (y - mu)^2) %*% A) %*% t(bread))[2, 2]))
}

qs_fine <- seq(0, 3, length.out = 121)
v_opt <- asym_cov(2)[2, 2]
eff_curve <- data.frame(q = qs_fine, eff = sapply(qs_fine, function(q) v_opt / asym_cov(q)[2, 2]))
qs <- seq(0, 3, by = 0.5); R <- 2000; z <- qnorm(0.975)
Y <- matrix(rgamma(R * n, shape = 1 / PHI, scale = rep(mu0 * PHI, each = R)), R, n)
sims <- do.call(rbind, lapply(qs, function(q) {
  fits <- lapply(1:R, function(r) fit(Y[r, ], q))
  b1 <- sapply(fits, function(f) f$b[2])
  data.frame(q = q, v = var(b1),
             cov_m = mean(abs(b1 - BETA[2]) <= z * sapply(fits, `[[`, "se_m")),
             cov_s = mean(abs(b1 - BETA[2]) <= z * sapply(fits, `[[`, "se_s")))
}))
sims$eff <- sims$v[sims$q == 2] / sims$v

p1 <- ggplot() +
  geom_vline(xintercept = 2, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(data = eff_curve, aes(q, eff, colour = "asymptotic"), linewidth = 0.8) +
  geom_point(data = sims, aes(q, eff, colour = "simulation, n = 200"), shape = 21, fill = "white", size = 2, stroke = 0.9) +
  scale_colour_manual(values = c(PALETTE[["blue"]], PALETTE[["blue"]]),
                      guide = guide_legend(override.aes = list(linetype = c(1, 0), shape = c(NA, 21)))) +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(x = expression("working variance power" ~ q ~ "in" ~ V(mu) == mu^q),
       y = expression("efficiency of" ~ hat(beta)[1] ~ "relative to" ~ q == 2), title = "Efficiency") +
  theme_blog()
cov_df <- rbind(data.frame(q = sims$q, cov = sims$cov_m, kind = "model-based"),
                data.frame(q = sims$q, cov = sims$cov_s, kind = "sandwich"))
p2 <- ggplot(cov_df, aes(q, cov, colour = kind, shape = kind)) +
  geom_hline(yintercept = 0.95, colour = PALETTE[["ink"]], linewidth = 0.35) +
  geom_vline(xintercept = 2, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  geom_line(linewidth = 0.8) + geom_point(fill = "white", size = 2, stroke = 0.9) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue")])) + scale_shape_manual(values = c(21, 22)) +
  coord_cartesian(ylim = c(0.7, 1)) +
  labs(x = "working variance power q", y = "coverage of nominal 95% interval",
       title = expression("Confidence intervals for" ~ beta[1])) +
  theme_blog()
save_fig(p1 + p2, here, "working-variance", width = 7.8, height = 3.7)
