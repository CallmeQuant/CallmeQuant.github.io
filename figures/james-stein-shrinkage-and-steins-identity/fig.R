# James-Stein estimation. Figure 1: exact risk relative to the MLE. Figure 2: one draw with p = 30.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
set.seed(1961)

js_risk <- function(p, lam) { k <- 0:1999; p - (p - 2)^2 * sum(dpois(k, lam / 2) / (p - 2 + 2 * k)) }

# ---- Figure 1 ----
r <- seq(0, 12, length.out = 241)
curves <- do.call(rbind, lapply(c(3, 5, 10, 30), function(p)
  data.frame(r = r, risk = sapply(r, function(t) js_risk(p, t^2)) / p, p = paste("p =", p))))
curves$p <- factor(curves$p, levels = paste("p =", c(3, 5, 10, 30)))
p <- 10; R <- 40000; rs <- 0:12
pp <- data.frame(r = rs, risk = sapply(rs, function(t) {
  theta <- c(t, rep(0, p - 1))
  X <- matrix(rnorm(R * p), R, p) + rep(theta, each = R)
  shrink <- pmax(0, 1 - (p - 2) / rowSums(X^2))
  mean(rowSums((shrink * X - rep(theta, each = R))^2)) / p
}))
g1 <- ggplot(curves, aes(r, risk, colour = p)) +
  geom_hline(yintercept = 1, colour = PALETTE[["ink"]], linetype = "dashed", linewidth = 0.35) +
  annotate("text", x = 11.9, y = 1.03, label = "MLE", hjust = 1, size = 3.2, colour = PALETTE[["ink"]]) +
  geom_line(linewidth = 0.8) +
  geom_point(data = pp, aes(r, risk), inherit.aes = FALSE, shape = 21, fill = "white", colour = CYCLE[3], size = 1.8, stroke = 0.9) +
  scale_colour_manual(values = CYCLE[1:4]) +
  coord_cartesian(xlim = c(0, 12), ylim = c(0, 1.08)) +
  labs(x = expression("||" * theta * "||"), y = expression("risk relative to the MLE," ~ R(theta) / p),
       caption = "circles: positive-part James-Stein, p = 10 (simulated)") +
  theme_blog()
save_fig(g1, here, "js-risk", width = 6.6, height = 3.8)

# ---- Figure 2 ----
p <- 30
theta <- 2 + rnorm(p, 0, 0.8)
X <- theta + rnorm(p)
js0 <- (1 - (p - 2) / sum(X^2)) * X
xbar <- mean(X); jsm <- xbar + (1 - (p - 3) / sum((X - xbar)^2)) * (X - xbar)
loss <- function(e) sum((e - theta)^2)
d <- rbind(data.frame(theta = theta, est = X, kind = sprintf("MLE: loss %.1f", loss(X))),
           data.frame(theta = theta, est = js0, kind = sprintf("James-Stein toward 0: loss %.1f", loss(js0))),
           data.frame(theta = theta, est = jsm, kind = sprintf("James-Stein toward mean: loss %.1f", loss(jsm))))
d$kind <- factor(d$kind, levels = unique(d$kind))
g2 <- ggplot(d, aes(theta, est)) +
  geom_abline(slope = 1, intercept = 0, colour = PALETTE[["grey"]], linewidth = 0.35) +
  geom_hline(yintercept = xbar, colour = PALETTE[["blue"]], linetype = "dotted", linewidth = 0.35) +
  geom_segment(data = data.frame(theta = theta, a = X, b = jsm), aes(x = theta, xend = theta, y = a, yend = b),
               colour = PALETTE[["light"]], linewidth = 0.5) +
  geom_point(aes(colour = kind, shape = kind), size = 1.8, fill = "white", stroke = 0.8) +
  scale_colour_manual(values = unname(PALETTE[c("grey", "gold", "blue")])) +
  scale_shape_manual(values = c(16, 24, 15)) +
  guides(colour = guide_legend(ncol = 1), shape = guide_legend(ncol = 1)) +
  labs(x = expression("true mean" ~ theta[i]), y = "estimate", title = "One draw, p = 30; dotted line: mean of X") +
  theme_blog()
save_fig(g2, here, "js-one-draw", width = 6.6, height = 4.2)
