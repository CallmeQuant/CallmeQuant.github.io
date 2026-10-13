# Timing NumPy, eager JAX and jit-compiled JAX: plots the timings measured by fig.py (data.csv).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

d <- read.csv(file.path(here, "data.csv"), comment.char = "#")
e1 <- subset(d, experiment == "elementwise")
m1 <- c(numpy = "NumPy", jax_eager = "JAX, op by op", jax_jit = "JAX, jit (compiled)",
        jax_jit_first_call = "jit, first call (trace + compile)")
e1$lab <- factor(m1[e1$method], m1)
p1 <- ggplot(e1, aes(size, seconds, colour = lab, linetype = lab)) + geom_line(linewidth = 0.55) + geom_point(size = 1.3) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(4e-6, 200)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "gold", "blue", "grey")])) +
  scale_linetype_manual(values = c(1, 1, 1, 2)) +
  labs(x = "array length n", y = "seconds per call", title = "Elementwise expression and sum") +
  theme_blog() + theme(legend.position = "top", legend.direction = "vertical", legend.text = element_text(size = 8))

e2 <- subset(d, experiment == "recursion")
m2 <- c(python_loop = "Python loop, floats", numpy_loop = "Python loop, NumPy scalars",
        jax_eager_loop = "Python loop, JAX op by op", jax_scan_jit = "lax.scan under jit")
e2$lab <- factor(m2[e2$method], m2)
p2 <- ggplot(e2, aes(size, seconds, colour = lab)) + geom_line(linewidth = 0.55) + geom_point(size = 1.3) +
  scale_x_log10(labels = scales::label_math(10^.x, format = log10)) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(4e-6, 200)) +
  scale_colour_manual(values = unname(PALETTE[c("red", "violet", "gold", "blue")])) +
  labs(x = "number of time steps T", y = "seconds per call",
       title = expression("Recursion" ~ x[t] == tanh(0.9 * x[t - 1]) + e[t])) +
  theme_blog() + theme(legend.position = "top", legend.direction = "vertical", legend.text = element_text(size = 8))
save_fig(p1 + p2, here, "timings", width = 7.8, height = 4.4)
