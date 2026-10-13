# Bayesian model averaging over all 2^6 linear regressions with Zellner's g-prior (g = n):
# inclusion probabilities, the model-averaged posterior of beta_6, and out-of-sample log scores.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

n <- 60; p <- 6
beta <- c(0.6, 0, 0.4, 0, 0, 0.25)
R <- matrix(0.3, p, p); diag(R) <- 1; R[1, 2] <- R[2, 1] <- 0.8
U <- chol(R)
MODELS <- as.matrix(expand.grid(rep(list(c(FALSE, TRUE)), p)))[, p:1]  # rows are subsets

simulate <- function(m) {
  X <- matrix(rnorm(m * p), m, p) %*% U
  list(X = X, y = drop(X %*% beta) + rnorm(m))
}

fit <- function(X, y, g) {
  m <- length(y); xbar <- colMeans(X); ybar <- mean(y)
  Xc <- sweep(X, 2, xbar); yc <- y - ybar; tss <- sum(yc^2)
  out <- lapply(seq_len(nrow(MODELS)), function(i) {
    gam <- MODELS[i, ]; k <- sum(gam)
    if (k > 0) {
      Xk <- Xc[, gam, drop = FALSE]; XtXinv <- solve(crossprod(Xk))
      bhat <- drop(XtXinv %*% crossprod(Xk, yc)); R2 <- sum(yc * (Xk %*% bhat)) / tss
    } else { XtXinv <- matrix(0, 0, 0); bhat <- numeric(0); R2 <- 0 }
    list(gam = gam, logml = 0.5 * (m - 1 - k) * log1p(g) - 0.5 * (m - 1) * log1p(g * (1 - R2)),
         mean = g / (1 + g) * bhat, S = tss * (1 - g / (1 + g) * R2), XtXinv = XtXinv, shrink = g / (1 + g))
  })
  lw <- sapply(out, `[[`, "logml"); w <- exp(lw - max(lw))
  list(out = out, w = w / sum(w), xbar = xbar, ybar = ybar, m = m)
}

ldt <- function(x, df, loc, scale) dt((x - loc) / scale, df, log = TRUE) - log(scale)

log_pred <- function(f, Xnew, ynew) {
  t(sapply(f$out, function(o) {
    Xc <- sweep(Xnew, 2, f$xbar)[, o$gam, drop = FALSE]
    loc <- f$ybar + if (any(o$gam)) drop(Xc %*% o$mean) else 0
    quad <- if (any(o$gam)) rowSums((Xc %*% o$XtXinv) * Xc) else 0
    ldt(ynew, f$m - 1, loc, sqrt(o$S / (f$m - 1) * (1 + 1 / f$m + o$shrink * quad)))
  }))
}

set.seed(7)
d <- simulate(n)
f <- fit(d$X, d$y, g = n)
pip <- drop(t(MODELS) %*% f$w)

grid <- seq(-0.3, 0.8, length.out = 800); j <- 6
cont <- rep(0, length(grid))
for (i in seq_along(f$out)) {
  o <- f$out[[i]]
  if (o$gam[j]) {
    pos <- match(j, which(o$gam))
    sd <- sqrt(o$S / (n - 1) * o$shrink * o$XtXinv[pos, pos])
    cont <- cont + f$w[i] * exp(ldt(grid, n - 1, o$mean[pos], sd))
  }
}
full <- f$out[[length(f$out)]]
dens_full <- exp(ldt(grid, n - 1, full$mean[j], sqrt(full$S / (n - 1) * full$shrink * full$XtXinv[j, j])))

d1 <- data.frame(x = factor(paste0("x[", 1:p, "]"), levels = paste0("x[", 1:p, "]")), pip = pip,
                 active = ifelse(beta != 0, "nonzero", "zero"))
p1 <- ggplot(d1, aes(x, pip, fill = active)) + geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.2f", pip)), vjust = -0.5, size = 3.1, family = "serif") +
  scale_fill_manual(values = unname(PALETTE[c("blue", "grey")]), guide = "none") +
  scale_x_discrete(labels = scales::label_parse()) +
  coord_cartesian(ylim = c(0, 1.1), expand = FALSE) +
  labs(x = NULL, y = "posterior inclusion probability", title = "Inclusion probabilities") +
  theme_blog() + theme(panel.grid.major.x = element_blank())

d2 <- rbind(data.frame(b = grid, dens = cont, what = "BMA, continuous part"),
            data.frame(b = grid, dens = dens_full, what = "full model only"))
p2 <- ggplot(d2, aes(b, dens, colour = what, linetype = what)) +
  geom_vline(xintercept = beta[j], colour = PALETTE[["grey"]], linetype = "dotted") +
  geom_line(linewidth = 0.8) +
  annotate("segment", x = 0, xend = 0, y = 0, yend = 1 - pip[j], colour = PALETTE[["blue"]], linewidth = 1.6) +
  annotate("point", x = 0, y = 1 - pip[j], colour = PALETTE[["blue"]], size = 2) +
  annotate("text", x = -0.27, y = 2.7, hjust = 0, size = 3.2, family = "serif",
           label = sprintf("mass %.2f at 0", 1 - pip[j])) +
  annotate("segment", x = -0.15, xend = 0, y = 2.55, yend = 1 - pip[j], linewidth = 0.25) +
  scale_colour_manual(values = unname(PALETTE[c("blue", "red")])) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  coord_cartesian(xlim = c(-0.3, 0.8), ylim = c(0, 3.6), expand = FALSE) +
  labs(x = expression(beta[6]), y = "posterior density", title = expression("Model-averaged posterior of" ~ beta[6])) +
  theme_blog() + theme(legend.position = c(0.75, 0.88), legend.justification = c(0.5, 0.5))
save_fig(p1 + p2 + plot_layout(widths = c(1, 1.25)), here, "bma-posterior", width = 7.6, height = 3.3)

reps <- 300; m_test <- 200
set.seed(11)
diffs <- t(replicate(reps, {
  tr <- simulate(n); te <- simulate(m_test)
  f <- fit(tr$X, tr$y, g = n)
  lp <- log_pred(f, te$X, te$y)
  top <- apply(lp, 2, max)
  bma <- mean(log(colSums(f$w * exp(sweep(lp, 2, top)))) + top)
  c(hpm = bma - mean(lp[which.max(f$w), ]), full = bma - mean(lp[nrow(lp), ]))
}))
d3 <- rbind(data.frame(d = diffs[, "hpm"], what = "BMA minus highest-probability model"),
            data.frame(d = diffs[, "full"], what = "BMA minus full model"))
d3$what <- factor(d3$what, levels = c("BMA minus highest-probability model", "BMA minus full model"))
lab <- do.call(rbind, lapply(split(d3, d3$what), function(s)
  data.frame(what = factor(as.character(s$what[1]), levels = levels(d3$what)), m = mean(s$d), txt = sprintf("mean %.3f\npositive in %.0f%%", mean(s$d), 100 * mean(s$d > 0)))))
p3 <- ggplot(d3, aes(d, fill = what)) +
  geom_histogram(breaks = seq(-0.10, 0.145, by = 0.005), colour = "white", linewidth = 0.2) +
  geom_vline(xintercept = 0, colour = PALETTE[["ink"]], linewidth = 0.35) +
  geom_vline(data = lab, aes(xintercept = m), colour = PALETTE[["red"]], linetype = "dashed") +
  geom_text(data = lab, aes(x = 0.145, y = Inf, label = txt), hjust = 1, vjust = 1.3, size = 3.1,
            family = "serif", inherit.aes = FALSE) +
  facet_wrap(~what) +
  scale_fill_manual(values = unname(PALETTE[c("blue", "green")]), guide = "none") +
  labs(x = "difference in mean log predictive density", y = "replications") +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p3, here, "bma-predictive", width = 7.6, height = 3.0)
