# ==========================================================================
# TEMA 06: DBSCAN
# Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# DBSCAN en R con dbscan y kNNdistplot
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

library(dbscan)

X <- as.matrix(iris[, 1:4])

# Calibración de eps: gráfico de distancias al k-ésimo vecino
kNNdistplot(X, k = 5)
abline(h = 0.6, col = "red", lty = 2)

# Ajuste del modelo
db <- dbscan(X, eps = 0.6, minPts = 6)
db

db$cluster       # etiquetas, el 0 corresponde al ruido (en R es 0, no -1)
table(db$cluster)

plot(X[, 3], X[, 4], col = db$cluster + 1L, pch = 16)


# Datos con formas irregulares
set.seed(42)
theta <- seq(0, 2*pi, length.out = 300)
r1 <- 2 + rnorm(300, 0, 0.4)
r2 <- 5 + rnorm(300, 0, 0.4)

circ1 <- cbind(r1*cos(theta), r1*sin(theta))
circ2 <- cbind(r2*cos(theta), r2*sin(theta))
ruido <- cbind(runif(100, -8, 8), runif(100, -8, 8))

Xirr <- rbind(circ1, circ2, ruido)
plot(Xirr, pch = 16, cex = 0.5)

kNNdistplot(Xirr, k = 8)
db2 <- dbscan(Xirr, eps = 0.7, minPts = 8)
plot(Xirr, col = db2$cluster + 1L, pch = 16, cex = 0.6)


# OPTICS, la versión que maneja densidades variables
op <- optics(Xirr, eps = 2, minPts = 8)
plot(op)
