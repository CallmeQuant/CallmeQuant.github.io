# Dirichlet process mixture of bivariate normals (known covariance sigma^2 I, base measure
# N(mu0, tau0^2 I)) fitted by collapsed Gibbs sampling (Neal's Algorithm 3): data and one posterior
# partition, trace of the number of clusters, and prior versus posterior of K.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(2000)
centers <- rbind(c(0, 0), c(3, 0), c(0, 3), c(3.5, 3.5))
props <- c(0.35, 0.30, 0.20, 0.15)
n <- 300; sigma <- 0.5
labels <- sample(4, n, replace = TRUE, prob = props)
X <- centers[labels, ] + sigma * matrix(rnorm(2 * n), n, 2)
alpha <- 1; tau0 <- 3; mu0 <- colMeans(X)

log_pred <- function(x, cnt, sm) {          # sm: K x 2 matrix of coordinate sums
  s2 <- 1 / (1 / tau0^2 + cnt / sigma^2)
  m <- s2 * (matrix(mu0 / tau0^2, length(cnt), 2, byrow = TRUE) + sm / sigma^2)
  v <- sigma^2 + s2
  -log(2 * pi * v) - rowSums((matrix(x, length(cnt), 2, byrow = TRUE) - m)^2) / (2 * v)
}

sweeps <- 400; burn <- 100
z <- rep(1, n); cnt <- n; sm <- matrix(colSums(X), 1, 2)
trace <- integer(sweeps)
for (it in 1:sweeps) {
  for (i in sample(n)) {
    k <- z[i]; cnt[k] <- cnt[k] - 1; sm[k, ] <- sm[k, ] - X[i, ]
    if (cnt[k] == 0) { cnt <- cnt[-k]; sm <- sm[-k, , drop = FALSE]; z[z > k] <- z[z > k] - 1 }
    lp <- c(log(cnt) + log_pred(X[i, ], cnt, sm), log(alpha) + log_pred(X[i, ], 0, matrix(0, 1, 2)))
    p <- exp(lp - max(lp))
    k <- sample(length(p), 1, prob = p)
    if (k > length(cnt)) { cnt <- c(cnt, 0); sm <- rbind(sm, c(0, 0)) }
    z[i] <- k; cnt[k] <- cnt[k] + 1; sm[k, ] <- sm[k, ] + X[i, ]
  }
  trace[it] <- length(cnt)
}

rank_by_size <- function(lab) { r <- rank(-tabulate(lab), ties.method = "first"); r[lab] }
pal <- rep(CYCLE, 5)
d1 <- rbind(data.frame(x1 = X[, 1], x2 = X[, 2], g = factor(rank_by_size(labels)), panel = "Data, coloured by true component"),
            data.frame(x1 = X[, 1], x2 = X[, 2], g = factor(rank_by_size(z)), panel = paste("Partition at sweep", sweeps)))
d1$panel <- factor(d1$panel, levels = unique(d1$panel))
p1 <- ggplot(d1, aes(x1, x2, colour = g)) + geom_point(size = 1, alpha = 0.85) +
  facet_wrap(~panel) + scale_colour_manual(values = pal, guide = "none") + coord_equal() +
  labs(x = expression(x[1]), y = expression(x[2])) + theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p1, here, "dpmm-partition", width = 7.6, height = 3.8)

pk <- 1
for (i in 1:n) { q <- alpha / (alpha + i - 1); pk <- c(pk * (1 - q), 0) + c(0, pk * q) }
ks <- 1:15
d2 <- data.frame(sweep = 1:sweeps, K = trace)
d3 <- rbind(data.frame(K = ks - 0.2, p = pk[ks + 1], what = "prior"),
            data.frame(K = ks + 0.2, p = sapply(ks, function(k) mean(trace[-(1:burn)] == k)), what = "posterior"))
p2 <- ggplot(d2, aes(sweep, K)) + geom_step(colour = PALETTE[["blue"]], linewidth = 0.35, direction = "mid") +
  geom_vline(xintercept = burn, colour = PALETTE[["grey"]], linetype = "dashed") +
  annotate("text", x = burn + 6, y = 10.4, label = "burn-in ends", hjust = 0, size = 3, colour = PALETTE[["grey"]]) +
  coord_cartesian(xlim = c(0, sweeps), ylim = c(0, 11), expand = FALSE) +
  labs(x = "sweep", y = "number of clusters K", title = "Trace of the number of clusters") + theme_blog()
p3 <- ggplot(d3, aes(K, p, fill = what)) + geom_col(width = 0.4) +
  scale_fill_manual(values = c(prior = PALETTE[["grey"]], posterior = PALETTE[["blue"]])) +
  scale_x_continuous(breaks = c(1, 4, 7, 10, 13)) +
  coord_cartesian(xlim = c(0.3, 15.7), ylim = c(0, 1.12 * max(d3$p)), expand = FALSE) +
  labs(x = "K", y = "probability", title = "Prior and posterior of K") +
  theme_blog() + theme(legend.position = c(0.75, 0.85), legend.justification = c(0.5, 0.5),
                       panel.grid.major.x = element_blank())
save_fig(p2 + p3 + plot_layout(widths = c(1.4, 1)), here, "dpmm-clusters", width = 7.6, height = 3.1)
