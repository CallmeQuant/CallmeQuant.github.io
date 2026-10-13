# Probabilistic PCA: sample versus fitted eigenvalues, and EM convergence to the closed-form MLE.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))
suppressPackageStartupMessages(library(patchwork))

set.seed(1999)
n <- 200; p <- 20; q <- 3; sigma2 <- 1
W_true <- matrix(rnorm(p * q), p) %*% diag(c(3, 2, 1.4))
X <- matrix(rnorm(n * q), n) %*% t(W_true) + sqrt(sigma2) * matrix(rnorm(n * p), n)
Xc <- sweep(X, 2, colMeans(X)); S <- crossprod(Xc) / n
e <- eigen(S, symmetric = TRUE); lam <- e$values; U <- e$vectors
loglik <- function(W, s2) {
  C <- W %*% t(W) + s2 * diag(p)
  -0.5 * n * (p * log(2 * pi) + as.numeric(determinant(C)$modulus) + sum(diag(solve(C, S))))
}
s2_ml <- mean(lam[(q + 1):p])
W_ml <- U[, 1:q] %*% diag(sqrt(lam[1:q] - s2_ml))
ll_ml <- loglik(W_ml, s2_ml)

da <- data.frame(j = 1:p, lam = lam, model = c(lam[1:q], rep(s2_ml, p - q)))
pa <- ggplot(da) +
  geom_hline(yintercept = sigma2, colour = PALETTE[["red"]], linetype = "dotted") +
  geom_step(aes(j, model, colour = "eigenvalues of fitted C"), direction = "mid", linewidth = 0.8) +
  geom_point(aes(j, lam, colour = "eigenvalues of S"), size = 1.8) +
  scale_colour_manual(values = c("eigenvalues of S" = PALETTE[["ink"]], "eigenvalues of fitted C" = PALETTE[["blue"]])) +
  scale_y_log10() + scale_x_continuous(breaks = c(1, 5, 10, 15, 20)) +
  labs(x = expression("index" ~ j), y = "eigenvalue (log scale)",
       title = bquote("(a)" ~ p == .(p) * "," ~ q == .(q) * ":" ~ hat(sigma)^2 == .(sprintf("%.2f", s2_ml)))) +
  theme_blog() + theme(legend.position = c(0.98, 0.98), legend.justification = c(1, 1))

T <- 3000
P_true <- U[, 1:q] %*% t(U[, 1:q])
db <- NULL
for (i in 1:4) {
  W <- matrix(rnorm(p * q), p); s2 <- 5; gaps <- sub <- numeric(T)
  for (it in 1:T) {
    gaps[it] <- ll_ml - loglik(W, s2)
    Pw <- W %*% solve(crossprod(W), t(W)); sub[it] <- norm(Pw - P_true, "2")
    M <- crossprod(W) + s2 * diag(q); Minv <- solve(M); SW <- S %*% W
    W_new <- SW %*% solve(s2 * diag(q) + Minv %*% t(W) %*% SW)
    s2 <- sum(diag(S - SW %*% Minv %*% t(W_new))) / p
    W <- W_new
  }
  g_end <- if (any(gaps < 1e-10)) which(gaps < 1e-10)[1] else T
  s_end <- if (any(sub < 1e-13)) which(sub < 1e-13)[1] else T
  db <- rbind(db, data.frame(t = 1:g_end, v = gaps[1:g_end], run = i, what = "log-likelihood gap"),
              data.frame(t = 1:s_end, v = sub[1:s_end], run = i, what = "distance to principal subspace"))
}
db$what <- factor(db$what, levels = c("log-likelihood gap", "distance to principal subspace"))
pb <- ggplot(db, aes(t, v, colour = factor(run), linetype = what, group = interaction(run, what))) +
  geom_line(linewidth = 0.6) +
  scale_colour_manual(values = CYCLE[1:4], guide = "none") +
  scale_linetype_manual(values = c("solid", "dashed")) +
  scale_x_log10() + scale_y_log10(labels = scales::label_math(10^.x, format = log10)) +
  coord_cartesian(ylim = c(1e-14, 1e5)) +
  labs(x = "EM iteration t (log scale)", y = NULL, title = "(b) EM from four random starts") +
  theme_blog() + theme(legend.position = c(1, 0.02), legend.justification = c(1, 0), legend.text = element_text(size = 7.5))
save_fig(pa + pb, here, "ppca-fit", width = 7.8, height = 3.4)
