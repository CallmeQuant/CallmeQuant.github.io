# Control variates for E[log(1 + exp(Z))], Z ~ N(0.5, 1.5^2).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

mu <- 0.5; s <- 1.5
f <- function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))     # softplus

# Reference values by numerical integration.
th_int <- function(h) integrate(function(z) h(z) * dnorm(z, mu, s), -Inf, Inf, rel.tol = 1e-12)$value
theta <- th_int(f)
var_f <- th_int(function(z) (f(z) - theta)^2)
gfun <- list(function(z) z - mu, function(z) (z - mu)^2 - s^2)
resid_var <- sapply(0:2, function(k) {
  if (k == 0) return(var_f)
  Sgg <- outer(1:k, 1:k, Vectorize(function(i, j) th_int(function(z) gfun[[i]](z) * gfun[[j]](z))))
  Sgf <- sapply(1:k, function(i) th_int(function(z) gfun[[i]](z) * (f(z) - theta)))
  var_f - sum(Sgf * solve(Sgg, Sgf))
})

estimate <- function(z, k) {
  y <- f(z)
  if (k == 0) return(mean(y))
  X <- cbind(1, sapply(gfun[seq_len(k)], function(g) g(z)))
  qr.coef(qr(X), y)[1]
}

# (a) the regression picture for one sample
set.seed(5)
z <- rnorm(60, mu, s); d <- data.frame(g = z - mu, y = f(z))
fit <- lm(y ~ g, d); a <- coef(fit)[1]; b <- coef(fit)[2]
pts <- data.frame(g = c(mean(d$g), 0), y = c(mean(d$y), a),
                  lab = c(sprintf("plain mean %.3f", mean(d$y)), sprintf("line at g = 0: %.3f", a)))
pa <- ggplot(d, aes(g, y)) +
  geom_hline(yintercept = theta, linetype = 3, linewidth = 0.3) + geom_vline(xintercept = 0, linewidth = 0.25) +
  geom_point(colour = PALETTE[["grey"]], size = 1) +
  geom_abline(intercept = a, slope = b, colour = PALETTE[["blue"]], linewidth = 0.6) +
  geom_point(data = pts, aes(colour = lab, shape = lab), size = 2.6) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  scale_shape_manual(values = c(15, 16)) +
  annotate("text", x = 3.9, y = theta - 0.45, label = sprintf("theta == %.3f", theta), parse = TRUE, hjust = 1) +
  coord_cartesian(xlim = c(-4, 4), ylim = c(-0.3, 4.6)) +
  labs(x = expression("control" ~ g == Z - mu), y = expression(f(Z) == log(1 + e^Z)), title = "One sample, n = 60") +
  theme_blog() + theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

# (b) root mean squared error against n
ns <- c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000); R <- 2000
set.seed(6)
labs3 <- c("'plain Monte Carlo'", "'control' ~ Z - mu", "'controls' ~ Z - mu * ',' ~ (Z - mu)^2 - sigma^2")
emp <- do.call(rbind, lapply(0:2, function(k) data.frame(n = ns, k = labs3[k + 1],
  rmse = sapply(ns, function(n) sqrt(mean((replicate(R, estimate(rnorm(n, mu, s), k)) - theta)^2))))))
nn <- exp(seq(log(8), log(7000), length.out = 50))
thr <- do.call(rbind, lapply(0:2, function(k) data.frame(n = nn, k = labs3[k + 1], rmse = sqrt(resid_var[k + 1] / nn))))
emp$k <- factor(emp$k, labs3); thr$k <- factor(thr$k, labs3)
pb <- ggplot(thr, aes(n, rmse, colour = k)) + geom_line(linewidth = 0.5) + geom_point(data = emp, size = 1.5) +
  scale_x_log10() + scale_y_log10(limits = c(3e-4, 1.2)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "blue", "green")]), labels = scales::label_parse()) +
  labs(x = "sample size n", y = "root mean squared error",
       title = expression("Error against" ~ n ~ "(lines:" ~ sqrt(sigma[f]^2 * (1 - R^2) / n) * ")")) +
  theme_blog() + theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0), legend.text = element_text(size = 8))
save_fig(pa + pb + plot_layout(widths = c(1, 1.1)), here, "control-variates", width = 7.8, height = 3.7)
