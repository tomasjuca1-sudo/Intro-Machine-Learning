# ==========================================================================
# TEMA 04: K-MEDOIDES
# Clase 3
# ==========================================================================


# --------------------------------------------------------------------------
# PAM y distancia de Gower en R
# Origen: Equivalente agregado, no viene del material del curso. La función daisy con metric gower si aparece en el código de MDS del curso
# --------------------------------------------------------------------------

library(cluster)

# Caso euclidiano
Xs <- scale(iris[, 1:4])
pam_res <- pam(Xs, k = 3)

pam_res$medoids     # coordenadas de los medoides
pam_res$id.med      # índices de los medoides dentro de la muestra
pam_res$clustering  # etiquetas
pam_res$objective   # valor de la función objetivo

plot(Xs[, 3], Xs[, 4], col = pam_res$clustering, pch = 16)
points(pam_res$medoids[, 3], pam_res$medoids[, 4], pch = 8, cex = 2)

# Silhouette de la solución
plot(silhouette(pam_res))


# Caso no euclidiano: datos mixtos con distancia de Gower
xx <- read.table("sponge.txt", header = TRUE, sep = ",")
dis_gower <- daisy(xx, metric = "gower")

pam_gower <- pam(dis_gower, k = 3, diss = TRUE)
pam_gower$id.med
pam_gower$clustering


# CLARA: versión de PAM para muestras grandes
clara_res <- clara(Xs, k = 3, samples = 50)
clara_res$medoids
