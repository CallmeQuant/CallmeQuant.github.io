# Ridge regression with p > n: coefficient paths and the leave-one-out shortcut.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1)
n <- 30; p <- 60; rho <- 0.6
Sigma <- rho^abs(outer(1:p, 1:p, "-"))
X <- matrix(rnorm(n * p), n, p) %*% t(chol(Sigma))
beta <- c(2.0, -1.5, 1.0, 0.8, -0.6, rep(0, p - 5))
y <- drop(X %*% beta + 2.0 * rnorm(n))

s <- svd(X); U <- s$u; d <- s$d; V <- s$v
Uty <- drop(crossprod(U, y))
lams <- 10^seq(-3, 4, length.out = 200)
paths <- t(sapply(lams, function(l) drop(V %*% (d / (d^2 + l) * Uty))))
bmin <- drop(V %*% (Uty / d))

hat <- function(l) U %*% diag(d^2 / (d^2 + l)) %*% t(U)
loo_short <- function(l) { H <- hat(l); r <- y - H %*% y; mean((r / (1 - diag(H)))^2) }
gcv <- function(l) { H <- hat(l); r <- y - H %*% y; mean(r^2) / (1 - sum(diag(H)) / n)^2 }
loo_brute <- function(l) mean(sapply(1:n, function(i) {
  b <- solve(crossprod(X[-i, ]) + l * diag(p), crossprod(X[-i, ], y[-i]))
  y[i] - sum(X[i, ] * b)
})^2)

dp <- data.frame(lambda = rep(lams, p), b = as.vector(paths), j = rep(1:p, each = length(lams)))
dp$grp <- ifelse(dp$j <= 5, paste0("j", dp$j), "other")
cols <- c(setNames(CYCLE[1:5], paste0("j", 1:5)), other = PALETTE[["light"]])
pa <- ggplot() +
  geom_line(data = subset(dp, grp == "other"), aes(lambda, b, group = j), colour = PALETTE[["light"]], linewidth = 0.35) +
  geom_line(data = subset(dp, grp != "other"), aes(lambda, b, group = j, colour = grp), linewidth = 0.8) +
  geom_point(data = data.frame(lambda = lams[1] * 0.55, b = bmin[1:5], grp = paste0("j", 1:5)),
             aes(lambda, b, colour = grp), shape = 17, size = 2.4) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  labs(x = expression(lambda ~ "(log scale)"), y = expression(hat(beta)[list(lambda, j)]),
       title = expression("(a) Paths; triangles mark" ~ (X^"+" * y)[j])) +
  theme_blog()

lg <- 10^seq(-2, 3, length.out = 80); lb <- 10^seq(-2, 3, length.out = 11)
dl <- rbind(data.frame(lambda = lg, v = sapply(lg, loo_short), what = "LOO (shortcut)"),
            data.frame(lambda = lg, v = sapply(lg, gcv), what = "GCV"))
db <- data.frame(lambda = lb, v = sapply(lb, loo_brute), what = "LOO (n refits)")
pb <- ggplot() +
  geom_line(data = dl, aes(lambda, v, colour = what, linetype = what), linewidth = 0.8) +
  geom_point(data = db, aes(lambda, v, colour = what), shape = 21, fill = "white", size = 2, stroke = 0.9) +
  scale_colour_manual(values = c("LOO (shortcut)" = PALETTE[["blue"]], "GCV" = PALETTE[["gold"]],
                                 "LOO (n refits)" = PALETTE[["red"]]),
                      breaks = c("LOO (shortcut)", "LOO (n refits)", "GCV")) +
  guides(colour = guide_legend(override.aes = list(shape = c(NA, 21, NA), linetype = c(1, 0, 2),
                                                   fill = "white"))) +
  scale_linetype_manual(values = c("LOO (shortcut)" = "solid", "GCV" = "dashed"), guide = "none") +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  labs(x = expression(lambda ~ "(log scale)"), y = "prediction error estimate",
       title = "(b) Leave-one-out error") +
  theme_blog() + theme(legend.position = c(0.02, 0.04), legend.justification = c(0, 0))
save_fig(pa + pb, here, "paths-and-loo", width = 7.6, height = 3.4)
