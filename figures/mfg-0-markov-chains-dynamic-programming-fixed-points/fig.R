# Figures for "Markov chains, dynamic programming and fixed points" (mean-field games, Part 0).
# Packages: ggplot2, patchwork, scales
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(scales) })

# ------------------------------------------------------------------ the example MDP
d <- 11; Tm <- 14; eta <- 0.3; kappa <- 0.5; lam <- 0.15
X <- 0:(d - 1); A <- c(-1, 0, 1)
clip <- function(z) pmin(pmax(z, 0), d - 1)
p <- array(0, dim = c(3, d, d))                 # p[a, x, y] = p(y | x, a), 1-based indices
for (k in 1:3) for (x in X) {
  for (s in list(c(0, 1 - eta), c(-1, eta / 2), c(1, eta / 2))) {
    y <- clip(x + A[k] + s[1]); p[k, x + 1, y + 1] <- p[k, x + 1, y + 1] + s[2]
  }
}
cost <- outer(lam * (X - 3)^2, kappa * abs(A), "+")   # cost[x, a]
g <- (X - 8)^2

V <- matrix(0, Tm + 1, d); V[Tm + 1, ] <- g
policy <- matrix(0L, Tm, d)
for (t in Tm:1) {
  Q <- sapply(1:3, function(k) cost[, k] + p[k, , ] %*% V[t + 1, ])   # d x 3
  policy[t, ] <- apply(Q, 1, which.min); V[t, ] <- apply(Q, 1, min)
}
m0 <- numeric(d); m0[2:6] <- dbinom(0:4, 4, 0.5)
mu <- matrix(0, Tm + 1, d); mu[1, ] <- m0
for (t in 1:Tm) {
  P <- t(sapply(1:d, function(x) p[policy[t, x], x, ]))
  mu[t + 1, ] <- mu[t, ] %*% P
}
J <- sum(sapply(1:Tm, function(t) sum(mu[t, ] * cost[cbind(1:d, policy[t, ])]))) + sum(mu[Tm + 1, ] * g)
stopifnot(abs(J - sum(m0 * V[1, ])) < 1e-9)

# ------------------------------------------------------------------ Figure 2: backward and forward
pol <- expand.grid(x = X, t = 0:(Tm - 1))
pol$a <- factor(A[policy[cbind(pol$t + 1, pol$x + 1)]], levels = c(-1, 0, 1), labels = c("down", "stay", "up"))
arrows_df <- subset(pol, a != "stay")
arrows_df$dir <- ifelse(arrows_df$a == "up", 1, -1)
pa <- ggplot(pol, aes(t, x, fill = a)) +
  geom_tile() +
  geom_segment(data = arrows_df, aes(x = t, xend = t, y = x - 0.25 * dir, yend = x + 0.25 * dir),
               colour = "white", linewidth = 0.35, arrow = arrow(length = unit(0.07, "cm"), type = "closed"),
               inherit.aes = FALSE) +
  geom_point(data = subset(pol, a == "stay"), aes(t, x), colour = PALETTE[["ink"]], size = 0.4, inherit.aes = FALSE) +
  scale_fill_manual(values = c(down = PALETTE[["red"]], stay = "#eef1f6", up = PALETTE[["blue"]]), name = NULL) +
  scale_x_continuous(breaks = seq(0, Tm - 1, 2), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, d - 1, 2), expand = c(0, 0)) +
  labs(x = "time t", y = "state x", title = expression("(a) optimal action " * varphi[t](x))) +
  theme_blog() + theme(panel.grid.major = element_blank(), legend.position = "none")

flow <- expand.grid(x = X, t = 0:Tm)
flow$m <- mu[cbind(flow$t + 1, flow$x + 1)]
pb <- ggplot(flow, aes(t, x, fill = m)) +
  geom_tile() +
  scale_fill_gradientn(colours = c("#ffffff", "#c9d5ef", PALETTE[["blue"]], "#16264d"), name = NULL) +
  scale_x_continuous(breaks = seq(0, Tm, 2), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, d - 1, 2), expand = c(0, 0)) +
  labs(x = "time t", y = NULL, title = expression("(b) state distribution " * mu[t](x))) +
  theme_blog() + theme(panel.grid.major = element_blank(), legend.position = "right",
                       legend.key.width = unit(0.6, "lines"), legend.key.height = unit(2.2, "lines"))
save_fig(pa + pb, here, "backward-forward", width = 7.6, height = 3.6)

# ------------------------------------------------------------------ Figure 1: empirical measures
set.seed(20261101)
muT <- mu[Tm + 1, ]
bars <- do.call(rbind, lapply(c(40, 400), function(N)
  data.frame(x = X, prob = as.vector(rmultinom(1, N, muT)) / N, N = paste0("N = ", N))))
bars$N <- factor(bars$N, levels = c("N = 40", "N = 400"))
p2a <- ggplot(bars, aes(x, prob, fill = N)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.78) +
  geom_point(data = data.frame(x = X, prob = muT), aes(x, prob), inherit.aes = FALSE,
             colour = PALETTE[["ink"]], size = 1.8) +
  scale_fill_manual(values = unname(PALETTE[c("gold", "green")]),
                    labels = c(expression(bar(mu)[T]^N * ",  N = 40"), expression(bar(mu)[T]^N * ",  N = 400"))) +
  scale_x_continuous(breaks = seq(0, d - 1, 2)) +
  coord_cartesian(ylim = c(0, 0.7)) +
  labs(x = "state x", y = NULL, title = "(a) one draw of the empirical measure") +
  annotate("text", x = 10, y = 0.68, label = "dots: law at time T", hjust = 1, size = 3.2, colour = PALETTE[["ink"]]) +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

Ns <- unique(round(10^seq(1, 4, length.out = 13)))
reps <- 4000
err <- sapply(Ns, function(N) mean(colSums(abs(rmultinom(reps, N, muT) / N - muT))))
clt <- sqrt(2 / pi) * sum(sqrt(muT * (1 - muT)))
lines_df <- rbind(data.frame(N = Ns, e = sqrt((d - 1) / Ns), what = "bound sqrt((d-1)/N)"),
                  data.frame(N = Ns, e = clt / sqrt(Ns), what = "CLT approximation"))
p2b <- ggplot() +
  geom_line(data = lines_df, aes(N, e, colour = what, linetype = what), linewidth = 0.7) +
  geom_point(data = data.frame(N = Ns, e = err), aes(N, e, colour = "Monte Carlo"), size = 1.8) +
  scale_x_log10(labels = label_number(big.mark = ",")) + scale_y_log10() +
  scale_colour_manual(values = c("bound sqrt((d-1)/N)" = PALETTE[["red"]], "CLT approximation" = PALETTE[["grey"]],
                                 "Monte Carlo" = PALETTE[["blue"]]), breaks = c("Monte Carlo", "bound sqrt((d-1)/N)", "CLT approximation"),
                      labels = c(expression("Monte Carlo " * E * "||" * bar(mu)[T]^N - mu[T] * "||"[1]),
                                 expression("bound " * sqrt((d - 1) / N)), "CLT approximation")) +
  scale_linetype_manual(values = c("bound sqrt((d-1)/N)" = "solid", "CLT approximation" = "dashed"), guide = "none") +
  labs(x = "number of chains N", y = NULL, title = expression("(b) " * L^1 * " error at time T")) +
  guides(colour = guide_legend(override.aes = list(linetype = c(0, 1, 2), shape = c(16, NA, NA)))) +
  theme_blog() + theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0))
save_fig(p2a + p2b, here, "empirical-measure", width = 8.2, height = 3.6)

# ------------------------------------------------------------------ Figure 3: fixed points
s <- 1 / 3
br <- rbind(data.frame(p = c(0, 0, 1, 1), q = c(0, s, s, 1), who = "BR1(q)"),
            data.frame(p = c(0, s, s, 1), q = c(0, 0, 1, 1), who = "BR2(p)"))
eq <- data.frame(p = c(0, s, 1), q = c(0, s, 1))
p3a <- ggplot(br, aes(p, q, colour = who, linetype = who)) +
  geom_path(linewidth = 1) +
  geom_point(data = eq, aes(p, q), inherit.aes = FALSE, shape = 21, size = 3.2, fill = "white",
             colour = PALETTE[["ink"]], stroke = 0.9) +
  annotate("text", x = 0.47, y = 0.2, label = "(1/3, 1/3)", size = 3.4) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]),
                      labels = c(expression(BR[1](q)), expression(BR[2](p)))) +
  scale_linetype_manual(values = c("solid", "dashed"), labels = c(expression(BR[1](q)), expression(BR[2](p)))) +
  coord_equal(xlim = c(-0.04, 1.04), ylim = c(-0.04, 1.04)) +
  labs(x = "p = probability that player 1 plays L", y = "q = probability that player 2 plays L",
       title = "(a) three Nash equilibria") +
  theme_blog() + theme(legend.position = c(0.03, 0.9), legend.justification = c(0, 1))

Fsegs <- data.frame(x = c(0, 0.5, 0.5), xend = c(0.5, 1, 0.5), y = c(0.8, 0.2, 0.2), yend = c(0.8, 0.2, 0.8))
p3b <- ggplot() +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = PALETTE[["grey"]]) +
  geom_segment(data = Fsegs, aes(x, y, xend = xend, yend = yend, colour = "F(1/2) = [0.2, 0.8]"), linewidth = 1) +
  geom_point(data = data.frame(x = c(0.5, 0.5), y = c(0.2, 0.8)), aes(x, y, colour = "G(1/2) = {0.2, 0.8}"), size = 2.4) +
  geom_point(aes(0.5, 0.5), shape = 21, size = 3.2, fill = "white", colour = PALETTE[["ink"]], stroke = 0.9) +
  annotate("text", x = 0.55, y = 0.47, label = "fixed point of F", hjust = 0, size = 3.3) +
  scale_colour_manual(values = c("F(1/2) = [0.2, 0.8]" = PALETTE[["blue"]], "G(1/2) = {0.2, 0.8}" = PALETTE[["red"]])) +
  guides(colour = guide_legend(override.aes = list(linetype = c(1, 0), shape = c(NA, 16)))) +
  coord_equal(xlim = c(-0.02, 1.02), ylim = c(-0.02, 1.02)) +
  labs(x = "x", y = "F(x), G(x)", title = "(b) why the values must be convex") +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))
save_fig(p3a + p3b, here, "fixed-points", width = 7.8, height = 4)

cat(sprintf("J = %.6f, <m0, V0> = %.6f\n", J, sum(m0 * V[1, ])))
