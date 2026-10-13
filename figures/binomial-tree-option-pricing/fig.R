# Binomial (Cox-Ross-Rubinstein) option pricing: a four-period American-put tree with
# early-exercise nodes, CRR call error against Black-Scholes, American versus European put.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(dplyr) })

# ---------------------------------------------------------------- tree
S0 <- 100; K <- 100; u <- 1.15; d <- 1 / u; R <- 1.01; N <- 4
q <- (R - d) / (u - d)
nodes <- expand.grid(n = 0:N, k = 0:N) |> filter(k <= n) |>
  mutate(S = S0 * u^k * d^(n - k), V = NA_real_, ex = FALSE)
idx <- function(n, k) which(nodes$n == n & nodes$k == k)
for (k in 0:N) nodes$V[idx(N, k)] <- max(K - nodes$S[idx(N, k)], 0)
for (n in (N - 1):0) for (k in 0:n) {
  i <- idx(n, k)
  cont <- (q * nodes$V[idx(n + 1, k + 1)] + (1 - q) * nodes$V[idx(n + 1, k)]) / R
  stop <- max(K - nodes$S[i], 0)
  nodes$V[i] <- max(cont, stop); nodes$ex[i] <- stop > cont
}
nodes <- nodes |> mutate(x = n, y = 2 * k - n,
                         type = ifelse(ex, "early exercise optimal", "continue (or expiry)"))
edges <- bind_rows(lapply(seq_len(nrow(nodes)), function(i) {
  if (nodes$n[i] == N) return(NULL)
  data.frame(x = nodes$x[i], y = nodes$y[i], xend = nodes$x[i] + 1, yend = nodes$y[i] + c(-1, 1))
}))
p_tree <- ggplot() +
  geom_segment(data = edges, aes(x, y, xend = xend, yend = yend), colour = PALETTE[["light"]], linewidth = 0.8) +
  geom_point(data = nodes, aes(x, y, fill = type), shape = 21, size = 15, stroke = 0.8,
             colour = ifelse(nodes$ex, PALETTE[["red"]], PALETTE[["blue"]])) +
  geom_text(data = nodes, aes(x, y + 0.2, label = sprintf("%.1f", S),
                              colour = type), size = 3, family = "serif") +
  geom_text(data = nodes, aes(x, y - 0.22, label = sprintf("%.2f", V),
                              colour = type), size = 3, family = "serif", fontface = "bold") +
  scale_fill_manual(values = c("early exercise optimal" = PALETTE[["red"]], "continue (or expiry)" = "white")) +
  scale_colour_manual(values = c("early exercise optimal" = "white", "continue (or expiry)" = PALETTE[["ink"]]),
                      guide = "none") +
  scale_x_continuous(breaks = 0:N, limits = c(-0.5, N + 0.5)) +
  scale_y_continuous(limits = c(-N - 0.8, N + 0.8)) +
  labs(x = "period n", y = NULL,
       title = "American put, K = 100, u = 1.15, d = 1/u, R = 1.01: node shows S (top) and V (bottom)") +
  theme_blog() +
  theme(axis.text.y = element_blank(), axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid.major = element_blank(), legend.position = c(0.02, 0.98), legend.justification = c(0, 1)) +
  guides(fill = guide_legend(override.aes = list(size = 4, colour = c(PALETTE[["blue"]], PALETTE[["red"]]))))
save_fig(p_tree, here, "tree", width = 7.2, height = 4.0)

# ------------------------------------------------------------ convergence
crr <- function(S0, K, r, sigma, T, N, kind = "call", american = FALSE) {
  dt <- T / N; u <- exp(sigma * sqrt(dt)); d <- 1 / u; R <- exp(r * dt)
  q <- (R - d) / (u - d)
  pay <- if (kind == "call") function(s) pmax(s - K, 0) else function(s) pmax(K - s, 0)
  sapply(S0, function(s0) {
    V <- pay(s0 * u^(0:N) * d^(N - 0:N))
    for (n in (N - 1):0) {
      V <- (q * V[2:(n + 2)] + (1 - q) * V[1:(n + 1)]) / R
      if (american) V <- pmax(V, pay(s0 * u^(0:n) * d^(n - 0:n)))
    }
    V[1]
  })
}
bs <- function(S0, K, r, sigma, T, kind = "call") {
  d1 <- (log(S0 / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T)); d2 <- d1 - sigma * sqrt(T)
  if (kind == "call") S0 * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
  else K * exp(-r * T) * pnorm(-d2) - S0 * pnorm(-d1)
}

r <- 0.05; sigma <- 0.2; T <- 1
Ns <- 5:300
err <- sapply(Ns, function(n) crr(100, 105, r, sigma, T, n)) - bs(100, 105, r, sigma, T)
de <- data.frame(N = Ns, y = Ns * err, parity = ifelse(Ns %% 2 == 0, "N even", "N odd"))
p1 <- ggplot(de, aes(N, y, colour = parity)) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = PALETTE[["ink"]]) +
  geom_line(linewidth = 0.3) + geom_point(size = 0.6) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "gold")])) +
  coord_cartesian(xlim = c(0, 300), ylim = c(-2.6, 3.1)) +
  labs(x = "number of periods N", y = NULL,
       title = expression("call: " * N * (C[N] - C[BS]) * ",  " * S[0] * " = 100,  K = 105")) +
  theme_blog() + theme(legend.position = c(1, 1), legend.justification = c(1, 1), legend.direction = "horizontal")

sig2 <- 0.3; Kp <- 100; Nam <- 500
s <- seq(50, 150, length.out = 201)
am <- crr(s, Kp, r, sig2, T, Nam, kind = "put", american = TRUE)
eu <- bs(s, Kp, r, sig2, T, kind = "put")
intr <- pmax(Kp - s, 0)
s_star <- max(s[am - intr < 1e-6])
dp <- bind_rows(data.frame(s = s, v = intr, what = "payoff (K - S0)+"),
                data.frame(s = s, v = eu, what = "European (Black-Scholes)"),
                data.frame(s = s, v = am, what = "American (CRR, N = 500)"))
dp$what <- factor(dp$what, levels = unique(dp$what))
p2 <- ggplot(dp, aes(s, v, colour = what)) + geom_line(linewidth = 0.7) +
  geom_vline(xintercept = s_star, linetype = "dotted", colour = PALETTE[["ink"]]) +
  annotate("text", x = s_star - 2, y = 8, label = sprintf("S* = %.1f", s_star), hjust = 1, size = 3.2) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "blue", "red")])) +
  coord_cartesian(xlim = c(50, 150), ylim = c(0, 52)) +
  labs(x = expression("spot price " * S[0]), y = NULL, title = expression("put value, K = 100, " * sigma == 0.3)) +
  theme_blog() + theme(legend.position = c(1, 1), legend.justification = c(1, 1))
save_fig(p1 + p2, here, "convergence", width = 7.8, height = 3.5)
