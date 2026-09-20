# ==========================================================================
# TEMA 02: K-MEANS
# Clase 2, Clase 3
# ==========================================================================


# --------------------------------------------------------------------------
# K-Means en R con escalamiento y codo
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Datos
X <- iris[, 1:4]

# Escalar es fundamental si las unidades difieren
Xs <- scale(X)

# K-Means con 3 grupos y 25 inicializaciones aleatorias
set.seed(0)
km <- kmeans(Xs, centers = 3, nstart = 25)

km$centers        # centroides (una fila por grupo)
km$cluster        # etiquetas de cada observación
km$tot.withinss   # WSS total = SSE
km$betweenss      # BSS
km$totss          # SST = BSS + WSS

plot(Xs[, 3], Xs[, 4], col = km$cluster, pch = 16)
points(km$centers[, 3], km$centers[, 4], col = 1:3, pch = 8, cex = 2)


# Método del codo
sse <- rep(0, 10)
for (k in 1:10) {
  sse[k] <- kmeans(Xs, centers = k, nstart = 25)$tot.withinss
}
plot(1:10, sse, type = "b", xlab = "K", ylab = "SSE (within)")
