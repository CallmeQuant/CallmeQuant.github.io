# Seed-to-seed variability: random initialization in a nonconvex fit, random train/test splits,
# and the optimism of reporting the best of k seeds.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

# ---- Figure 1: frequency estimation by gradient descent from random starting values
set.seed(101)
n <- 100
x <- runif(n, 0, 10)
y <- sin(1.3 * x) + rnorm(n, 0, 0.3)
loss <- function(w) sapply(w, function(v) mean((y - sin(v * x))^2))
grad <- function(w) sapply(w, function(v) -2 * mean((y - sin(v * x)) * cos(v * x) * x))
set.seed(7)
w0 <- runif(300, 0.2, 3); w <- w0
for (it in 1:3000) w <- w - 0.004 * grad(w)
ws <- seq(0.05, 3.2, length.out = 2000)
ends <- as.data.frame(table(round(w, 2))); ends$w <- as.numeric(as.character(ends$Var1)); ends$L <- loss(ends$w)
best <- ws[which.min(loss(ws))]
hit <- abs(w - best) < 0.02
pa <- ggplot(data.frame(w = ws, L = loss(ws)), aes(w, L)) + geom_line(colour = PALETTE[["ink"]], linewidth = 0.5) +
  geom_point(data = ends, aes(size = Freq), colour = PALETTE[["red"]], alpha = 0.8) +
  scale_size_area(max_size = 6, guide = "none") +
  coord_cartesian(xlim = c(0.05, 3.2), ylim = c(0, 1.25)) +
  labs(x = expression("frequency" ~ omega), y = expression("least-squares criterion" ~ L[n](omega)),
       title = "Nonconvex least squares, 300 random starts") + theme_blog()
pb <- ggplot(data.frame(w0 = w0, w = w, g = ifelse(hit, sprintf("reaches global minimum (%.0f%%)", 100 * mean(hit)),
                                                   "stops at a local minimum")), aes(w0, w, colour = g)) +
  geom_point(size = 0.9) + scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  coord_cartesian(xlim = c(0.1, 3.1), ylim = c(0, 3.3)) +
  labs(x = expression("starting value" ~ omega^(0) ~ "(set by the seed)"), y = expression("final value" ~ hat(omega)),
       title = "Output as a function of the seed") +
  theme_blog() + theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0), legend.text = element_text(size = 8))
save_fig(pa + pb + plot_layout(widths = c(1.35, 1)), here, "nonconvex-starts", width = 7.8, height = 3.4)

# ---- Figure 2: test R^2 across random splits, and best-of-k optimism
set.seed(202)
n <- 60; p <- 8
X <- matrix(rnorm(n * p), n)
beta <- c(1, -0.8, 0.6, 0.5, 0, 0, 0.3, 0)
yv <- drop(X %*% beta) + rnorm(n, 0, 1.2)
test_r2 <- function(seed) {
  set.seed(seed); idx <- sample(n); tr <- idx[1:45]; te <- idx[46:n]
  fit <- lm.fit(cbind(1, X[tr, ]), yv[tr])
  pred <- drop(cbind(1, X[te, ]) %*% fit$coefficients)
  1 - sum((yv[te] - pred)^2) / sum((yv[te] - mean(yv[te]))^2)
}
r2 <- sapply(0:4999, test_r2); mu <- mean(r2); sdv <- sd(r2); r42 <- test_r2(42)
pc <- ggplot(data.frame(r2 = r2), aes(r2)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60, fill = PALETTE[["light"]], colour = "white", linewidth = 0.2) +
  geom_vline(aes(xintercept = mu, colour = sprintf("mean over seeds %.2f", mu), linetype = sprintf("mean over seeds %.2f", mu))) +
  geom_vline(aes(xintercept = r42, colour = sprintf("seed 42: %.2f", r42), linetype = sprintf("seed 42: %.2f", r42))) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) + scale_linetype_manual(values = c(1, 2)) +
  coord_cartesian(xlim = c(-0.6, 1)) +
  labs(x = expression("test" ~ R^2 ~ "(45 training, 15 test points)"), y = "density",
       title = sprintf("One data set, 5000 split seeds (sd %.2f)", sdv)) +
  theme_blog() + theme(legend.position = c(0.02, 0.99), legend.justification = c(0, 1), legend.text = element_text(size = 8))
ks <- unique(round(exp(seq(log(1), log(200), length.out = 25))))
set.seed(303)
emax <- sapply(ks, function(k) mean(apply(matrix(sample(r2, 4000 * k, replace = TRUE), 4000), 1, max)))
lv <- c("expected best of k seeds", "mu + sigma sqrt(2 log k)", "mu: what a new split delivers")
dd <- rbind(data.frame(k = ks, v = emax, s = lv[1]), data.frame(k = ks, v = mu + sdv * sqrt(2 * log(ks)), s = lv[2]),
            data.frame(k = ks, v = mu, s = lv[3]))
dd$s <- factor(dd$s, lv)
pd <- ggplot(dd, aes(k, v, colour = s, linetype = s)) + geom_line(linewidth = 0.55) +
  geom_point(data = subset(dd, s == lv[1]), size = 1.2) +
  geom_hline(yintercept = 1, linetype = 3, linewidth = 0.3) +
  annotate("text", x = 1.05, y = 1.03, label = "R^2 <= 1", parse = TRUE, hjust = 0, vjust = 0, size = 3) +
  scale_x_log10() + coord_cartesian(ylim = c(-0.1, 1.3)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "grey", "blue")]),
                      labels = expression("expected best of" ~ k ~ "seeds", mu + sigma * sqrt(2 * log ~ k), mu * ":" ~ "what a new split delivers")) +
  scale_linetype_manual(values = c(1, 2, 1),
                        labels = expression("expected best of" ~ k ~ "seeds", mu + sigma * sqrt(2 * log ~ k), mu * ":" ~ "what a new split delivers")) +
  labs(x = "number of seeds tried, k", y = expression("reported test" ~ R^2), title = "Optimism from choosing the seed") +
  theme_blog() + theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0), legend.background = element_rect(fill = "white", colour = NA), legend.text = element_text(size = 8))
save_fig(pc + pd, here, "split-seeds", width = 7.8, height = 4)
