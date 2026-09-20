# ==========================================================================
# TEMA 08: LOF E ISOLATION FOREST
# Clase 6
# ==========================================================================


# --------------------------------------------------------------------------
# LOF e Isolation Forest en R
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Local Outlier Factor
library(dbscan)

X <- as.matrix(iris[, 1:4])

# k es el número de vecinos. Devuelve el LOF directo (no negado como en sklearn)
puntajes_lof <- lof(X, minPts = 5)

# LOF cerca de 1 = normal. LOF muy mayor que 1 = outlier
summary(puntajes_lof)
which(puntajes_lof > 1.5)

plot(X[, 3], X[, 4], pch = 16,
     cex = puntajes_lof,          # tamaño proporcional al LOF
     col = ifelse(puntajes_lof > 1.5, "red", "black"))


# Isolation Forest
library(isotree)

modelo_if <- isolation.forest(as.data.frame(X), ntrees = 50, nthreads = 1)

# En isotree el score va de 0 a 1: cerca de 1 es outlier
puntajes_if <- predict(modelo_if, as.data.frame(X))

summary(puntajes_if)
which(puntajes_if > 0.6)

plot(X[, 3], X[, 4], pch = 16,
     col = ifelse(puntajes_if > 0.6, "red", "black"))


# Alternativa con solitude
# library(solitude)
# iso <- isolationForest$new(num_trees = 50)
# iso$fit(as.data.frame(X))
# pred <- iso$predict(as.data.frame(X))
# pred$anomaly_score
