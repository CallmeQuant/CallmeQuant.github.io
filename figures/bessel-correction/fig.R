# Mean squared error of SS/d for d = n-1, n, n+1 and the MSE-optimal divisor, relative to d = n-1
# (note: Bessel's correction). Exact formulas.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

mse <- function(d, n, kappa) {
  ess2 <- (n - 1)^2 * (kappa / n - (n - 3) / (n * (n - 1)) + 1)
  ess2 / d^2 - 2 * (n - 1) / d + 1
}

n <- 2:30
cases <- data.frame(
  kappa = c(1, 3, 9),
  title = c("kappa == 1 ~ '(symmetric' %+-% '1)'", "kappa == 3 ~ '(normal)'", "kappa == 9 ~ '(exponential)'")
)
d <- do.call(rbind, lapply(seq_len(nrow(cases)), function(i) {
  k <- cases$kappa[i]
  base <- mse(n - 1, n, k)
  dstar <- n + 1 + (n - 1) * (k - 3) / n
  rbind(
    data.frame(n = n, ratio = mse(n, n, k) / base, divisor = "d == n", case = cases$title[i]),
    data.frame(n = n, ratio = mse(n + 1, n, k) / base, divisor = "d == n + 1", case = cases$title[i]),
    data.frame(n = n, ratio = mse(dstar, n, k) / base, divisor = "d == d^'*'", case = cases$title[i])
  )
}))
d$case <- factor(d$case, levels = cases$title)
d$divisor <- factor(d$divisor, levels = c("d == n", "d == n + 1", "d == d^'*'"))

p <- ggplot(d, aes(n, ratio, colour = divisor, linetype = divisor, linewidth = divisor)) +
  geom_hline(yintercept = 1, colour = PALETTE[["grey"]], linetype = "dashed", linewidth = 0.4) +
  geom_line() +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red", "ink")]), labels = function(x) parse(text = x)) +
  scale_linetype_manual(values = c("solid", "solid", "dotted"), labels = function(x) parse(text = x)) +
  scale_linewidth_manual(values = c(0.8, 0.8, 0.6), labels = function(x) parse(text = x)) +
  facet_wrap(~case, nrow = 1, labeller = label_parsed) +
  coord_cartesian(xlim = c(2, 30), ylim = c(0, 3.2)) +
  labs(x = "sample size n", y = "MSE(d) / MSE(n - 1)") +
  theme_blog() +
  theme(strip.text = element_text(hjust = 0))
save_fig(p, here, "mse-by-divisor", width = 8.2, height = 3.3)
