# ELBO and chi-square upper bound (CUBO_2) for a skew-normal posterior: optimal Gaussian fits, the
# evidence sandwich, and Monte Carlo behaviour of the CUBO estimate.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork) })

ALPHA <- 3; LOG_EV <- log(0.5)
log_joint <- function(z) pnorm(ALPHA * z, log.p = TRUE) + dnorm(z, log = TRUE)
elbo <- function(m, s) integrate(function(t) dnorm(t, m, s) * (log_joint(t) - dnorm(t, m, s, log = TRUE)),
                                 m - 12 * s, m + 12 * s, subdivisions = 400)$value
cubo2 <- function(m, s) {
  if (s^2 < 0.5 || (s^2 == 0.5 && m <= 0)) return(Inf)
  f <- function(t) exp(2 * log_joint(t) - dnorm(t, m, s, log = TRUE) - 2 * LOG_EV)
  LOG_EV + 0.5 * log(integrate(f, -Inf, m, subdivisions = 400)$value + integrate(f, m, Inf, subdivisions = 400)$value)
}
oe <- optim(c(0.75, log(0.6)), function(v) -elbo(v[1], exp(v[2])), control = list(reltol = 1e-12))
oc <- optim(c(0.75, log(0.75)), function(v) cubo2(v[1], exp(v[2])), control = list(reltol = 1e-12))
m_e <- oe$par[1]; s_e <- exp(oe$par[2]); m_c <- oc$par[1]; s_c <- exp(oc$par[2])
cat(sprintf("ELBO opt m=%.4f s2=%.4f; CUBO opt m=%.4f s2=%.4f\n", m_e, s_e^2, m_c, s_c^2))

z <- seq(-2.2, 3.4, length.out = 600)
labs1 <- c("posterior", "argmax ELBO", "argmin CUBO_2")
d1 <- rbind(data.frame(z = z, y = exp(log_joint(z) - LOG_EV), what = labs1[1]),
            data.frame(z = z, y = dnorm(z, m_e, s_e), what = labs1[2]),
            data.frame(z = z, y = dnorm(z, m_c, s_c), what = labs1[3]))
d1$what <- factor(d1$what, levels = labs1)
p1 <- ggplot(d1, aes(z, y, colour = what)) +
  geom_area(data = subset(d1, what == labs1[1]), fill = PALETTE[["light"]], colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 0.8) + scale_colour_manual(values = unname(PALETTE[c("ink", "blue", "red")]),
                      labels = c(expression("posterior" ~ pi), expression(arg ~ max ~ ELBO), expression(arg ~ min ~ CUBO[2]))) +
  coord_cartesian(xlim = c(-2.2, 3.4), ylim = c(0, 0.85), expand = FALSE) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(x = expression(z), y = "density", title = "Posterior and Gaussian fits") + theme_blog()

sg <- seq(0.45, 1.3, length.out = 70)
el <- sapply(sg, function(s) -optimize(function(m) -elbo(m, s), c(0.2, 1.3))$objective)
sgc <- c(sqrt(0.5), seq(sqrt(0.5) + 1e-4, 1.3, length.out = 70))
cu <- sapply(sgc, function(s) optimize(function(m) cubo2(m, s), c(0.05, 1.5))$objective)
d2 <- rbind(data.frame(s = sg, v = el, what = "max_m ELBO"), data.frame(s = sgc, v = cu, what = "min_m CUBO_2"))
d2$what <- factor(d2$what, levels = c("max_m ELBO", "min_m CUBO_2"))
p2 <- ggplot(d2, aes(s, v, colour = what)) +
  annotate("rect", xmin = 0.45, xmax = sqrt(0.5), ymin = -Inf, ymax = Inf, fill = PALETTE[["light"]]) +
  annotate("text", x = 0.465, y = -0.6, label = "atop(CUBO[2] == +infinity, 'for' ~ s^2 < 1/2)", parse = TRUE, hjust = 0, vjust = 1,
           size = 3, colour = PALETTE[["ink"]]) +
  geom_hline(yintercept = LOG_EV, linetype = "dashed", colour = PALETTE[["ink"]], linewidth = 0.4) +
  annotate("text", x = 1.29, y = LOG_EV + 0.004, label = "log ~ italic(p)(italic(y))", parse = TRUE, hjust = 1, vjust = 0, size = 3) +
  geom_line(linewidth = 0.8) +
  annotate("point", x = c(s_e, s_c), y = c(elbo(m_e, s_e), cubo2(m_c, s_c)), colour = unname(PALETTE[c("blue", "red")])) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]),
                      labels = c(expression(max[m] ~ ELBO), expression(min[m] ~ CUBO[2]))) +
  coord_cartesian(xlim = c(0.45, 1.3), ylim = c(-0.95, -0.56), expand = FALSE) +
  guides(colour = guide_legend(nrow = 2)) +
  labs(x = expression("scale" ~ s ~ "of" ~ q == N(m, s^2)), y = "nats", title = "The evidence sandwich") + theme_blog()
save_fig(p1 + p2, here, "fits-and-sandwich", width = 7.8, height = 3.6)

# Figure 2
set.seed(2017)
S_list <- c(10, 30, 100, 300, 1000, 3000, 10000); reps <- 2000
cubo_hat <- function(m, s, S) vapply(seq_len(reps), function(k) {
  zz <- rnorm(S, m, s); lw <- 2 * (log_joint(zz) - dnorm(zz, m, s, log = TRUE)); mx <- max(lw)
  0.5 * (mx + log(mean(exp(lw - mx))))
}, numeric(1))
d3 <- do.call(rbind, lapply(list(list(m_c, s_c, sprintf("q == arg ~ min ~ CUBO[2] ~ (s^2 == %.3f)", s_c^2)),
                                 list(m_e, s_e, sprintf("q == arg ~ max ~ ELBO ~ (s^2 == %.3f)", s_e^2))), function(cs) {
  qs <- t(sapply(S_list, function(S) quantile(cubo_hat(cs[[1]], cs[[2]], S), c(0.1, 0.5, 0.9))))
  data.frame(S = S_list, lo = qs[, 1], med = qs[, 2], hi = qs[, 3], case = cs[[3]])
}))
d3$case <- factor(d3$case, levels = unique(d3$case))
truth <- data.frame(case = factor(levels(d3$case)[1], levels = levels(d3$case)), v = cubo2(m_c, s_c))
p3 <- ggplot(d3, aes(S)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = case), alpha = 0.18) +
  geom_line(aes(y = med, colour = case), linewidth = 0.8) + geom_point(aes(y = med, colour = case), size = 1.4) +
  geom_hline(yintercept = LOG_EV, linetype = "dashed", colour = PALETTE[["ink"]], linewidth = 0.4) +
  geom_hline(data = truth, aes(yintercept = v), linetype = "dotted", colour = PALETTE[["red"]]) +
  scale_x_log10() + scale_colour_manual(values = unname(PALETTE[c("red", "blue")]), guide = "none") +
  scale_fill_manual(values = unname(PALETTE[c("red", "blue")]), guide = "none") +
  facet_wrap(~case, labeller = label_parsed) + coord_cartesian(ylim = c(-0.86, -0.42)) +
  labs(x = expression("number of draws" ~ S), y = expression("estimate of" ~ CUBO[2](q)),
       caption = expression("band: 10%-90% range; line: median; dashed:" ~ log ~ italic(p)(italic(y)) * "; dotted: true" ~ CUBO[2](q))) +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p3, here, "monte-carlo-cubo", width = 7.8, height = 3.4)
