# ==========================================================================
# TEMA 07: OUTLIERS: Z-SCORE Y MAHALANOBIS
# Clase 5
# ==========================================================================


# --------------------------------------------------------------------------
# Z-Score y Mahalanobis en R
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Z-Score
set.seed(1)
n <- 80
muestra <- rnorm(n, mean = 5, sd = 1.5)
muestra[n] <- 12.3          # contaminación

z <- scale(muestra)          # equivale a (x - mean)/sd
plot(z, type = "l", col = "red", lwd = 2)
which(abs(z) > 3)


# Mahalanobis
X <- as.matrix(iris[, 1:4])

medias <- colMeans(X)
S <- var(X)

d2 <- mahalanobis(X, center = medias, cov = S)   # devuelve d^2 directamente

p <- ncol(X)
umbral <- qchisq(0.975, df = p)

which(d2 > umbral)

plot(d2, pch = 16, col = ifelse(d2 > umbral, "red", "black"))
abline(h = umbral, col = "blue", lty = 2)

# QQ-plot contra la chi cuadrado
qqplot(qchisq(ppoints(nrow(X)), df = p), d2,
       xlab = "Cuantiles chi2", ylab = "Distancia de Mahalanobis al cuadrado")
abline(0, 1, col = "red")


# Versión robusta (MCD), no se enmascara con los propios outliers
library(MASS)
rob <- cov.rob(X, method = "mcd")
d2_rob <- mahalanobis(X, center = rob$center, cov = rob$cov)
which(d2_rob > umbral)
