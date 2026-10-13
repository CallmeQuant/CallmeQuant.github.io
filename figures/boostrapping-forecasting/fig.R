# Markov-chain bootstrap (Willemain, Smart and Schwarz) for intermittent demand on a simulated series
# (note: bootstrapping intermittent demand with a Markov chain).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(42)
p01 <- 0.10; p11 <- 0.60; size_mean <- 3
n_hist <- 730; L <- 12; B <- 20000

simulate <- function(n, state) {
  y <- integer(n)
  for (t in seq_len(n)) {
    state <- as.integer(runif(1) < if (state == 1) p11 else p01)
    if (state == 1) y[t] <- 1L + rpois(1, size_mean)
  }
  y
}

# The bootstrap as given in the note
fit_occurrence_chain <- function(y) {
  s <- as.integer(y > 0)
  n <- table(factor(head(s, -1), levels = 0:1), factor(tail(s, -1), levels = 0:1))
  if (any(rowSums(n) == 0)) stop("a state is never left")
  unclass(n / rowSums(n))
}
wss_bootstrap <- function(y, h, B = 5000) {
  P <- fit_occurrence_chain(y)
  sizes <- y[y > 0]
  out <- matrix(0, B, h)
  for (b in seq_len(B)) {
    state <- as.integer(y[length(y)] > 0)
    for (t in seq_len(h)) {
      state <- rbinom(1, 1, P[state + 1, 2])
      if (state == 1) {
        x <- sizes[sample.int(length(sizes), 1)]
        xj <- 1 + trunc(x + sqrt(x) * rnorm(1))
        out[b, t] <- if (xj > 0) xj else x
      }
    }
  }
  out
}

y <- simulate(n_hist, as.integer(runif(1) < p01 / (p01 + 1 - p11)))
last <- as.integer(y[n_hist] > 0)
truth <- replicate(B, sum(simulate(L, last)))
markov <- rowSums(wss_bootstrap(y, L, B))
iid <- rowSums(matrix(y[sample.int(n_hist, B * L, replace = TRUE)], B, L))

hist_df <- data.frame(t = (n_hist - 119):n_hist, y = y[(n_hist - 119):n_hist])
pa <- ggplot(subset(hist_df, y > 0), aes(t, y)) +
  geom_point(data = subset(hist_df, y == 0), aes(t, y), shape = "|", colour = PALETTE[["grey"]], size = 1.5) +
  geom_segment(aes(xend = t, yend = 0), colour = PALETTE[["blue"]], linewidth = 0.5) +
  geom_point(colour = PALETTE[["blue"]], size = 0.9) +
  labs(x = "period t", y = expression("demand" ~ y[t]), title = "(a) last 120 periods of the history") +
  theme_blog()

grid <- 0:60
ec <- function(d, lab) data.frame(x = grid, F = ecdf(d)(grid), what = lab)
cd <- rbind(ec(truth, "true distribution"), ec(markov, "Markov bootstrap"), ec(iid, "i.i.d. bootstrap"))
cd$what <- factor(cd$what, levels = c("true distribution", "Markov bootstrap", "i.i.d. bootstrap"))
pb <- ggplot(cd, aes(x, F, colour = what, linetype = what)) +
  geom_hline(yintercept = 0.95, colour = PALETTE[["grey"]], linetype = "dotted", linewidth = 0.4) +
  geom_step(linewidth = 0.6) +
  scale_colour_manual(values = unname(PALETTE[c("ink", "red", "gold")])) +
  scale_linetype_manual(values = c("solid", "solid", "dashed")) +
  coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.02)) +
  labs(x = "total demand over L = 12 periods", y = "distribution function", title = "(b) lead-time demand") +
  theme_blog() + theme(legend.position = c(0.97, 0.45), legend.justification = c(1, 0.5))

save_fig(pa + pb + plot_layout(widths = c(1.15, 1)), here, "lead-time-demand", width = 8.4, height = 3.3)
for (d in list(truth, markov, iid)) cat(sprintf("mean %.2f sd %.2f q95 %g\n", mean(d), sd(d), quantile(d, 0.95)))
