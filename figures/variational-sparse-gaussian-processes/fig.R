# Titsias's collapsed variational bound for sparse Gaussian process regression: the bound against the
# exact log marginal likelihood for nested inducing sets, and the effect of optimizing the inducing inputs.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

set.seed(2009)
ELL <- 0.6; SF <- 1; SN <- 0.25
kern <- function(a, b) SF^2 * exp(-0.5 * outer(a, b, "-")^2 / ELL^2)
f_true <- function(x) 0.8 * sin(2 * x) + 0.4 * cos(5 * x)
n <- 250; X <- sort(runif(n, 0, 8)); y <- f_true(X) + SN * rnorm(n)

lowrank <- function(Z) {
  m <- length(Z); Lu <- t(chol(kern(Z, Z) + 1e-8 * diag(m)))
  V <- forwardsolve(Lu, kern(Z, X))
  LB <- t(chol(diag(m) + V %*% t(V) / SN^2))
  cc <- forwardsolve(LB, V %*% y) / SN^2
  quad <- sum(y^2) / SN^2 - sum(cc^2)
  logdet <- 2 * sum(log(diag(LB))) + n * log(SN^2)
  c(ll = -0.5 * quad - 0.5 * logdet - 0.5 * n * log(2 * pi), tr = n * SF^2 - sum(V^2))
}
bound <- function(Z) { r <- lowrank(Z); r[["ll"]] - r[["tr"]] / (2 * SN^2) }
Rx <- chol(kern(X, X) + SN^2 * diag(n)); a <- forwardsolve(t(Rx), y)
log_ev <- -0.5 * sum(a^2) - sum(log(diag(Rx))) - 0.5 * n * log(2 * pi)

perm <- sample.int(n); ms <- 1:60
R <- t(sapply(ms, function(m) lowrank(X[perm[1:m]])))
d1 <- rbind(data.frame(m = ms, v = R[, "ll"], what = "log N(y | 0, Qff + s^2 I)"),
            data.frame(m = ms, v = R[, "ll"] - R[, "tr"] / (2 * SN^2), what = "bound L*(Z)"))
d1$what <- factor(d1$what, levels = unique(d1$what))
p1 <- ggplot(d1, aes(m, v, colour = what)) + geom_hline(yintercept = log_ev, linetype = "dashed", colour = PALETTE[["ink"]]) +
  geom_line(linewidth = 0.8) + scale_colour_manual(values = unname(PALETTE[c("gold", "blue")])) +
  coord_cartesian(ylim = c(-420, -30)) + guides(colour = guide_legend(ncol = 1)) +
  labs(x = "number of inducing inputs m", y = "nats", title = "Bound and approximate likelihood", caption = "dashed: exact log p(y)") + theme_blog()
d2 <- rbind(data.frame(m = ms, v = log_ev - (R[, "ll"] - R[, "tr"] / (2 * SN^2)), what = "gap log p(y) - L*"),
            data.frame(m = ms, v = R[, "tr"] / (2 * SN^2), what = "tr(Kff - Qff) / (2 s^2)"))
p2 <- ggplot(d2, aes(m, v, colour = what, linetype = what)) + geom_line(linewidth = 0.8) + scale_y_log10() +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) + scale_linetype_manual(values = c("solid", "dashed")) +
  guides(colour = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1)) +
  labs(x = "number of inducing inputs m", y = NULL, title = "Gap and trace penalty") + theme_blog()
save_fig(p1 + p2, here, "bound-vs-m", width = 7.8, height = 3.8)

predict <- function(xq, Z) {
  m <- length(Z); Kuu <- kern(Z, Z) + 1e-8 * diag(m); Kuf <- kern(Z, X); Ksu <- kern(xq, Z)
  A <- Kuu + Kuf %*% t(Kuf) / SN^2
  list(mean = drop(Ksu %*% solve(A, Kuf %*% y)) / SN^2,
       var = SF^2 - rowSums(Ksu * t(solve(Kuu, t(Ksu)))) + rowSums(Ksu * t(solve(A, t(Ksu)))))
}
m <- 16; Z0 <- seq(0.2, 2, length.out = m)
op <- optim(Z0, function(z) -bound(z), method = "L-BFGS-B", lower = 0, upper = 8)
Zopt <- sort(op$par)
xs <- seq(-0.5, 8.5, length.out = 500)
lab0 <- sprintf("initial inducing inputs: L* = %.1f", bound(Z0)); lab1 <- sprintf("after maximizing the bound over Z: L* = %.1f", bound(Zopt))
P <- rbind(with(predict(xs, Z0), data.frame(x = xs, mean = mean, sd = sqrt(pmax(var, 0)), panel = lab0)),
           with(predict(xs, Zopt), data.frame(x = xs, mean = mean, sd = sqrt(pmax(var, 0)), panel = lab1)))
P$panel <- factor(P$panel, levels = c(lab0, lab1))
Zd <- data.frame(x = c(Z0, Zopt), panel = factor(rep(c(lab0, lab1), each = m), levels = levels(P$panel)))
p3 <- ggplot(P, aes(x)) + geom_point(data = data.frame(x = X, y = y), aes(x, y), colour = PALETTE[["grey"]], size = 0.4, alpha = 0.6) +
  geom_ribbon(aes(ymin = mean - 1.96 * sd, ymax = mean + 1.96 * sd), fill = PALETTE[["blue"]], alpha = 0.18) +
  geom_line(aes(y = mean), colour = PALETTE[["blue"]], linewidth = 0.7) +
  geom_line(data = data.frame(x = xs, y = f_true(xs)), aes(x, y), linetype = "dashed", colour = PALETTE[["ink"]], linewidth = 0.35) +
  geom_point(data = Zd, aes(x, y = -2.15), shape = 17, colour = PALETTE[["red"]], size = 1.8) +
  facet_wrap(~panel, ncol = 1) + coord_cartesian(xlim = c(-0.5, 8.5), ylim = c(-2.3, 2.3), expand = FALSE) +
  labs(x = "x", y = "f(x)") + theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p3, here, "optimized-inducing-inputs", width = 6.8, height = 4.8)
