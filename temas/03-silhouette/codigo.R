# ==========================================================================
# TEMA 03: SILHOUETTE Y MÉTRICAS DE AJUSTE
# Clase 3, Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# Silhouette en R con el paquete cluster
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

library(cluster)

Xs <- scale(iris[, 1:4])
dis <- dist(Xs)            # matriz de distancias euclidianas

set.seed(0)
km <- kmeans(Xs, centers = 3, nstart = 25)

sil <- silhouette(km$cluster, dis)
summary(sil)               # promedio global y por cluster
plot(sil, col = 1:3)       # gráfico de silhouette

mean(sil[, 3])             # silhouette promedio


# Escoger K comparando el silhouette promedio
promedios <- rep(0, 10)
for (k in 2:10) {
  kmk <- kmeans(Xs, centers = k, nstart = 25)
  sk <- silhouette(kmk$cluster, dis)
  promedios[k] <- mean(sk[, 3])
}
plot(2:10, promedios[2:10], type = "b", xlab = "K", ylab = "Silhouette promedio")


# Calinski-Harabasz y Davies-Bouldin
library(fpc)
est <- cluster.stats(dis, km$cluster)
est$ch     # Calinski-Harabasz (se busca el máximo)

library(clusterSim)
index.DB(Xs, km$cluster)$DB   # Davies-Bouldin (se busca el mínimo)
