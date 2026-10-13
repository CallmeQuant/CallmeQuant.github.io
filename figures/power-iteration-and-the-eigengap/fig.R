# Power iteration on symmetric 200 x 200 matrices: angle error and Rayleigh quotient error.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(11)
n <- 200; K <- 150; FLOOR <- 1e-15
V <- qr.Q(qr(matrix(rnorm(n * n), n)))
x0 <- rnorm(n)
make <- function(r) {
  lam <- c(1, r, runif(n - 2, -0.5 * r, 0.5 * r))
  list(A = V %*% (lam * t(V)), lam = lam)
}
run <- function(A, lam) {
  v1 <- V[, 1]; x <- x0 / sqrt(sum(x0^2)); tn <- rq <- numeric(K + 1)
  for (k in 0:K) {
    cc <- sum(v1 * x)
    tn[k + 1] <- sqrt(sum((x - cc * v1)^2)) / abs(cc)
    rq[k + 1] <- abs(drop(t(x) %*% A %*% x) - lam[1])
    y <- A %*% x; x <- drop(y / sqrt(sum(y^2)))
  }
  list(tan = tn, rq = rq)
}
cut <- function(v) { i <- which(v < FLOOR)[1]; if (!is.na(i) && i < length(v)) v[(i + 1):length(v)] <- NA; v }

ratios <- c(0.5, 0.8, 0.9, 0.95, 0.99)
da <- do.call(rbind, lapply(ratios, function(r) { m <- make(r); data.frame(k = 0:K, v = cut(run(m$A, m$lam)$tan), r = factor(r, levels = ratios)) }))
lab <- do.call(rbind, lapply(split(da, da$r), function(d) { d <- d[!is.na(d$v), ]; d[nrow(d), ] }))
lab$v <- pmax(lab$v, 3e-16)
pa <- ggplot(da, aes(k, v, colour = r)) + geom_line(linewidth = 0.8, na.rm = TRUE) +
  geom_text(data = lab, aes(k + 4, v, label = r), hjust = 0, size = 3.2, family = "serif") +
  scale_colour_manual(values = CYCLE[1:5], guide = "none") +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-15, 3, 3)) +
  coord_cartesian(xlim = c(0, 170), ylim = c(1e-16, 1e3)) +
  labs(x = expression("iteration" ~ k), y = expression(tan ~ theta[k]),
       title = expression("(a) Angle to" ~ v[1] ~ "for" ~ lambda[2] / lambda[1] ~ "as labelled")) +
  theme_blog()

m <- make(0.9); o <- run(m$A, m$lam)
db <- rbind(data.frame(k = 0:K, v = cut(o$tan / sqrt(1 + o$tan^2)), w = "sin"),
            data.frame(k = 0:K, v = cut(o$rq), w = "rq"))
pb <- ggplot(db, aes(k, v, colour = w)) + geom_line(linewidth = 0.8, na.rm = TRUE) +
  scale_colour_manual(values = c(sin = PALETTE[["blue"]], rq = PALETTE[["red"]]), breaks = c("sin", "rq"),
                      labels = c(expression(sin ~ theta[k]), expression("|" * rho(x[k]) - lambda[1] * "|"))) +
  scale_y_log10(labels = scales::label_math(10^.x, format = log10), breaks = 10^seq(-16, 0, 2)) +
  coord_cartesian(ylim = c(1e-16, 3)) +
  labs(x = expression("iteration" ~ k), y = NULL,
       title = expression("(b)" ~ lambda[2] / lambda[1] == 0.9 * ": eigenvalue error squared")) +
  theme_blog() + theme(legend.position = c(0.03, 0.03), legend.justification = c(0, 0))
save_fig(pa + pb, here, "power-iteration-rates", width = 7.6, height = 3.5)
