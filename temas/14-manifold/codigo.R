# ==========================================================================
# TEMA 14: MANIFOLD LEARNING: ISOMAP, T-SNE, LLE
# Clase 8, Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# Isomap, LLE y t-SNE en R
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Datos tipo S-curve o swiss roll
library(dimRed)
library(vegan)
library(lle)
library(Rtsne)

# Generar un swiss roll (análogo a la S-curve)
set.seed(0)
n <- 1000
t <- (3*pi/2) * (1 + 2*runif(n))
h <- 21 * runif(n)
X <- cbind(t*cos(t), h, t*sin(t))
color <- t

library(scatterplot3d)
scatterplot3d(X, color = rainbow(n)[rank(color)], pch = 16)


# ISOMAP (paquete vegan)
dis <- dist(X)
iso <- isomap(dis, k = 10, ndim = 2)
plot(iso$points, col = rainbow(n)[rank(color)], pch = 16,
     main = "Isomap")


# LLE (paquete lle)
# calc_k ayuda a escoger el número de vecinos
# k_opt <- calc_k(X, m = 2, kmin = 5, kmax = 20)
res_lle <- lle(X, m = 2, k = 10)
plot(res_lle$Y, col = rainbow(n)[rank(color)], pch = 16,
     main = "LLE")


# t-SNE (paquete Rtsne)
res_tsne <- Rtsne(X, dims = 2, perplexity = 30, max_iter = 500, check_duplicates = FALSE)
plot(res_tsne$Y, col = rainbow(n)[rank(color)], pch = 16,
     main = "t-SNE")


# Comparación con PCA lineal
pr <- prcomp(X)
plot(pr$x[, 1:2], col = rainbow(n)[rank(color)], pch = 16,
     main = "PCA (no logra desenrollar)")
