# ==========================================================================
# TEMA 05: CLUSTERING JERÁRQUICO
# Clase 3, Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# hclust, dendrograma y corte del árbol
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

Xs <- scale(iris[, 1:4])
dis <- dist(Xs, method = "euclidean")

# Los cuatro linkages
hc_single   <- hclust(dis, method = "single")
hc_complete <- hclust(dis, method = "complete")
hc_average  <- hclust(dis, method = "average")
hc_ward     <- hclust(dis, method = "ward.D2")

par(mfrow = c(2, 2))
plot(hc_single,   main = "Single",   labels = FALSE)
plot(hc_complete, main = "Complete", labels = FALSE)
plot(hc_average,  main = "Average",  labels = FALSE)
plot(hc_ward,     main = "Ward",     labels = FALSE)
par(mfrow = c(1, 1))

# Cortar el árbol en K grupos
grupos <- cutree(hc_ward, k = 3)
table(grupos)
table(grupos, iris$Species)

# Dibujar los rectangulos del corte sobre el dendrograma
plot(hc_ward, labels = FALSE)
rect.hclust(hc_ward, k = 3, border = 2:4)

# Versión del paquete cluster (aglomerativo y divisivo)
library(cluster)
ag <- agnes(Xs, method = "ward")   # aglomerativo
di <- diana(Xs)                    # divisivo
plot(ag)
