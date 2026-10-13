# Kalman filter for a constant-velocity model with noisy position measurements.
# (a) true position, observations and filtered mean with 95% bands;
# (b) predicted and filtered variance of the position for the first 25 steps.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(MASS) })

set.seed(721)
T <- 60; dt <- 1; q <- 0.02; r <- 4
F <- matrix(c(1, 0, dt, 1), 2, 2)
Q <- q * matrix(c(dt^3 / 3, dt^2 / 2, dt^2 / 2, dt), 2, 2)
H <- matrix(c(1, 0), 1, 2)
m0 <- c(0, 0); P0 <- diag(c(4, 0.25))

z <- matrix(0, T + 1, 2); y <- numeric(T)
z[1, ] <- mvrnorm(1, m0, P0)
for (t in 2:(T + 1)) {
  z[t, ] <- F %*% z[t - 1, ] + mvrnorm(1, c(0, 0), Q)
  y[t - 1] <- z[t, 1] + rnorm(1, 0, sqrt(r))
}

m <- m0; P <- P0
mf <- numeric(T); vf <- numeric(T); vp <- numeric(T)
for (t in 1:T) {
  mp <- F %*% m; Pp <- F %*% P %*% t(F) + Q
  S <- H %*% Pp %*% t(H) + r
  K <- Pp %*% t(H) %*% solve(S)
  m <- mp + K %*% (y[t] - H %*% mp)
  P <- Pp - K %*% S %*% t(K)
  mf[t] <- m[1]; vf[t] <- P[1, 1]; vp[t] <- Pp[1, 1]
}

d <- data.frame(t = 1:T, y = y, truth = z[-1, 1], m = mf, lo = mf - 1.96 * sqrt(vf), hi = mf + 1.96 * sqrt(vf))
pa <- ggplot(d, aes(t)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = "95% band"), alpha = 0.25) +
  geom_point(aes(y = y, colour = "observation"), size = 1.1) +
  geom_line(aes(y = truth, colour = "true position"), linetype = "dashed", linewidth = 0.5) +
  geom_line(aes(y = m, colour = "filtered mean"), linewidth = 0.8) +
  scale_fill_manual(values = PALETTE[["blue"]], guide = "none") +
  scale_colour_manual(values = c("observation" = PALETTE[["grey"]], "true position" = PALETTE[["ink"]],
                                 "filtered mean" = PALETTE[["blue"]]),
                      breaks = c("observation", "true position", "filtered mean")) +
  guides(colour = guide_legend(override.aes = list(shape = c(16, NA, NA), linetype = c("blank", "dashed", "solid")))) +
  labs(x = "time t", y = "position", title = "(a) filtering the position") +
  theme_blog() + theme(legend.position = "bottom")

k <- 25
dv <- rbind(data.frame(t = 1:k, v = vp[1:k], what = "predicted"),
            data.frame(t = 1:k, v = vf[1:k], what = "filtered"))
zig <- data.frame(t = rep(1:k, each = 2), v = as.vector(rbind(vp[1:k], vf[1:k])))
pb <- ggplot() +
  geom_path(data = zig, aes(t, v), colour = PALETTE[["grey"]], linewidth = 0.4) +
  geom_point(data = dv, aes(t, v, colour = what), size = 1.6) +
  scale_colour_manual(values = c(predicted = PALETTE[["red"]], filtered = PALETTE[["blue"]]),
                      breaks = c("predicted", "filtered")) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(x = "time t", y = NULL, title = "(b) position variance") +
  theme_blog() + theme(legend.position = "bottom", legend.justification = "center")

save_fig(pa + pb + plot_layout(widths = c(1.6, 1)), here, "tracking", width = 7.8, height = 3.9)
