# Whitening a uniform parallelogram: ZCA, PCA and Cholesky whitening.
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
source(file.path(here, "..", "_style", "style.R"))

set.seed(404)
n <- 1500
S <- matrix(runif(2 * n, -sqrt(3), sqrt(3)), 2)
M <- matrix(c(1.2, 0.25, 0.7, 0.6), 2)          # [[1.2, 0.7], [0.25, 0.6]]
X <- M %*% S
Sigma <- M %*% t(M)
e <- eigen(Sigma, symmetric = TRUE); lam <- e$values; U <- e$vectors
U <- sweep(U, 2, sign(U[1, ]), "*")
W <- list(ZCA = U %*% diag(lam^-0.5) %*% t(U),
          PCA = diag(lam^-0.5) %*% t(U),
          Cholesky = solve(t(chol(Sigma))))
dist <- function(Wm) 2 + sum(diag(Sigma)) - 2 * sum(diag(Wm %*% Sigma))
sq <- rbind(c(-1, 1, 1, -1, -1), c(-1, -1, 1, 1, -1)) * sqrt(3)
labs <- c("'(a) data' ~ X",
          sprintf("'(b) ZCA:' ~ E * '||' * Z - X * '||'^2 == '%.2f'", dist(W$ZCA)),
          sprintf("'(c) PCA:' ~ E * '||' * Z - X * '||'^2 == '%.2f'", dist(W$PCA)),
          sprintf("'(d) Cholesky:' ~ E * '||' * Z - X * '||'^2 == '%.2f'", dist(W$Cholesky)))
mats <- c(list(diag(2)), W)
pts <- box <- crn <- NULL
for (k in 1:4) {
  Y <- mats[[k]] %*% X; B <- mats[[k]] %*% M %*% sq; C <- mats[[k]] %*% M %*% c(sqrt(3), sqrt(3))
  pts <- rbind(pts, data.frame(x = Y[1, ], y = Y[2, ], s = S[1, ], panel = labs[k]))
  box <- rbind(box, data.frame(x = B[1, ], y = B[2, ], panel = labs[k]))
  crn <- rbind(crn, data.frame(x = C[1], y = C[2], panel = labs[k]))
}
for (d in c("pts", "box", "crn")) assign(d, transform(get(d), panel = factor(panel, levels = labs)))
p <- ggplot() +
  geom_point(data = pts, aes(x, y, colour = s), size = 0.45, alpha = 0.8) +
  scale_colour_viridis_c(guide = "none") +
  geom_path(data = box, aes(x, y), colour = PALETTE[["ink"]], linewidth = 0.4) +
  geom_point(data = crn, aes(x, y), colour = PALETTE[["red"]], size = 2) +
  facet_wrap(~panel, ncol = 2, labeller = label_parsed) +
  coord_equal(xlim = c(-3.6, 3.6), ylim = c(-3.6, 3.6)) +
  labs(x = "first coordinate", y = "second coordinate") +
  theme_blog() + theme(strip.text = element_text(hjust = 0))
save_fig(p, here, "whitened-parallelogram", width = 6.4, height = 6.5)
