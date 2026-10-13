# Sample paths and sample autocorrelation functions for Gaussian white noise, an AR(1) process
# with phi = 0.7 and an MA(1) process with theta = 0.8.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(3)
n <- 200; H <- 20; burn <- 500; phi <- 0.7; theta <- 0.8
z <- rnorm(n + burn)
ar <- numeric(n + burn)
for (t in 2:(n + burn)) ar[t] <- phi * ar[t - 1] + z[t]
idx <- (burn + 1):(burn + n)
labs_ <- c("'white noise'", "'AR(1), ' * phi == 0.7", "'MA(1), ' * theta == 0.8")
xs <- list(z[idx], ar[idx], z[idx] + theta * z[idx - 1])
rhos <- list(rep(0, H), phi^(1:H), c(theta / (1 + theta^2), rep(0, H - 1)))

sample_acf <- function(x, H) {
  xc <- x - mean(x); g0 <- sum(xc^2) / length(x)
  sapply(1:H, function(h) sum(xc[(h + 1):length(x)] * xc[1:(length(x) - h)]) / length(x) / g0)
}
paths <- do.call(rbind, lapply(1:3, function(j) data.frame(t = 1:n, x = xs[[j]], s = labs_[j])))
acfs <- do.call(rbind, lapply(1:3, function(j)
  data.frame(h = 1:H, r = sample_acf(xs[[j]], H), rho = rhos[[j]], s = labs_[j])))
paths$s <- factor(paths$s, levels = labs_); acfs$s <- factor(acfs$s, levels = labs_)
band <- 1.96 / sqrt(n)

p1 <- ggplot(paths, aes(t, x)) + geom_line(colour = PALETTE[["blue"]], linewidth = 0.3) +
  facet_wrap(~s, nrow = 1, labeller = label_parsed) + labs(x = "t", y = expression(X[t])) + theme_blog()
p2 <- ggplot(acfs, aes(h)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -band, ymax = band, fill = PALETTE[["light"]]) +
  geom_hline(yintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.3) +
  geom_segment(aes(xend = h, y = 0, yend = r, colour = "sample"), linewidth = 0.8) +
  geom_point(aes(y = rho, colour = "true"), shape = 21, fill = "white", size = 1.6, stroke = 0.8) +
  scale_colour_manual(values = c(sample = PALETTE[["blue"]], true = PALETTE[["red"]]),
                      labels = c(sample = "sample autocorrelation", true = "true autocorrelation")) +
  guides(colour = guide_legend(override.aes = list(shape = c(NA, 21), linetype = c(1, 0)))) +
  scale_x_continuous(breaks = c(1, 5, 10, 15, 20)) +
  coord_cartesian(ylim = c(-0.35, 0.85)) +
  facet_wrap(~s, nrow = 1) + labs(x = "lag h", y = "autocorrelation") +
  theme_blog() + theme(strip.text = element_blank(), legend.position = "bottom")
save_fig(p1 / p2, here, "sample-acf", width = 8.2, height = 5)
