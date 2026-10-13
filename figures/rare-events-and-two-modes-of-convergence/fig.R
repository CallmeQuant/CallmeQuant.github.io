# Independent Bernoulli(p_n) indicators: times at which X_n = 1.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

set.seed(20191111)
N <- 10000; paths <- 10; n <- seq_len(N)
cases <- list(
  list(label = "p[n] == 1/n * ':' ~ sum(p[n]) == infinity * ',' ~ 'ones occur infinitely often'", p = 1 / n),
  list(label = "p[n] == 1/n^2 * ':' ~ sum(p[n]) < infinity * ',' ~ 'ones eventually stop'", p = 1 / n^2)
)
d <- do.call(rbind, lapply(cases, function(cs) do.call(rbind, lapply(seq_len(paths), function(j) {
  hits <- n[runif(N) < cs$p]
  data.frame(n = hits, path = j, case = cs$label)
}))))
d$case <- factor(d$case, levels = sapply(cases, `[[`, "label"))

p <- ggplot(d, aes(n, path, colour = case)) +
  geom_point(shape = "|", size = 3.2) +
  scale_x_log10(limits = c(0.8, N * 1.25), labels = scales::label_number(big.mark = ",")) +
  scale_y_continuous(breaks = c(1, 5, 10), limits = c(0.3, paths + 0.7)) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")]), guide = "none") +
  facet_wrap(~case, ncol = 1, labeller = label_parsed) +
  labs(x = "time n (log scale)", y = "path") +
  theme_blog() +
  theme(panel.grid.major.y = element_blank(), strip.text = element_text(hjust = 0))
save_fig(p, here, "indicator-paths", width = 6.8, height = 4.6)
