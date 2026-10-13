# Lasso sign recovery and the irrepresentable condition.
# Design: X1, X2 independent N(0,1); X3 has correlation c with each; beta* = (1, 1, 0); 2c is the
# irrepresentable quantity, so the condition holds for c = 0.4 and fails for c = 0.65.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages({ library(patchwork); library(glmnet) })
set.seed(2006)
SIGMA <- 1; B <- 1
design <- function(n, c, p_extra = 0) {
  x1 <- rnorm(n); x2 <- rnorm(n)
  x3 <- c * x1 + c * x2 + sqrt(1 - 2 * c^2) * rnorm(n)
  X <- cbind(x1, x2, x3, matrix(rnorm(n * p_extra), n, p_extra))
  list(X = X, y = B * (x1 + x2) + SIGMA * rnorm(n))
}
lasso <- function(X, y, lambda) glmnet(X, y, lambda = lambda, standardize = FALSE, intercept = FALSE, thresh = 1e-12)
case_lab <- c("0.4" = "c == 0.4 * ':' ~ max[j] ~ abs((Gamma * z[S])[j]) == 0.8",
              "0.65" = "c == 0.65 * ':' ~ max[j] ~ abs((Gamma * z[S])[j]) == 1.3")

# ---- Figure 1 ----
lams <- 10^seq(0.3, -3, length.out = 200)
paths <- do.call(rbind, lapply(c(0.4, 0.65), function(c) {
  d <- design(1000, c); fit <- lasso(d$X, d$y, lams)
  cf <- as.matrix(fit$beta)
  data.frame(lambda = rep(fit$lambda, each = 3), coef = as.vector(cf),
             which = rep(c("beta1", "beta2", "beta3 (true value 0)"), times = length(fit$lambda)),
             case = case_lab[[as.character(c)]])
}))
paths$case <- factor(paths$case, levels = unname(case_lab))
beta_labs <- c(expression(hat(beta)[1]), expression(hat(beta)[2]), expression(hat(beta)[3] ~ "(true value 0)"))
p1 <- ggplot(paths, aes(lambda, coef, colour = which, linetype = which)) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~case, labeller = label_parsed) +
  scale_x_continuous(trans = scales::compose_trans("log10", "reverse"), breaks = c(1, 0.1, 0.01, 0.001),
                     labels = c("1", "0.1", "0.01", "0.001")) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "green", "red")]), labels = beta_labs) +
  scale_linetype_manual(values = c("solid", "dashed", "solid"), labels = beta_labs) +
  labs(x = "lambda (log scale, decreasing)", y = "lasso coefficient") +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "lasso-paths", width = 7.8, height = 3.6)

# ---- Figure 2 ----
ns <- c(100, 200, 400, 800, 1600, 3200, 6400); R <- 200; p <- 50
truth <- sign(c(B, B, rep(0, p - 2)))
rec <- do.call(rbind, lapply(c(0.4, 0.65), function(c) do.call(rbind, lapply(ns, function(n) {
  lam <- 4 * SIGMA * sqrt(2 * log(p) / n)
  hits <- mean(replicate(R, {
    d <- design(n, c, p - 3)
    b <- as.vector(coef(lasso(d$X, d$y, lam)))[-1]
    b[abs(b) < 1e-10] <- 0
    all(sign(b) == truth)
  }))
  data.frame(n = n, prob = hits, case = case_lab[[as.character(c)]])
}))))
rec$case <- factor(rec$case, levels = unname(case_lab))
p2 <- ggplot(rec, aes(n, prob, colour = case, shape = case)) +
  geom_line(linewidth = 0.8) + geom_point(fill = "white", size = 2, stroke = 0.9) +
  scale_x_log10(breaks = ns) + scale_shape_manual(values = c(21, 22), labels = parse(text = unname(case_lab))) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), labels = parse(text = unname(case_lab))) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "sample size n (log scale)", y = expression(P * group("{", sign(hat(beta)) == sign(beta^"*"), "}"))) +
  theme_blog()
save_fig(p2, here, "sign-recovery", width = 6.4, height = 3.5)
