# ==========================================================================
# TEMA 17: VALIDACIÓN CRUZADA
# Clase 11, Clase 12
# ==========================================================================


# --------------------------------------------------------------------------
# Validación cruzada en R
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Partición simple
set.seed(0)
n <- nrow(datos)
idx <- sample(1:n, floor(0.67*n))
train <- datos[idx, ]
test  <- datos[-idx, ]


# Validación cruzada a mano con K folds
K <- 10
set.seed(0)
folds <- sample(rep(1:K, length.out = n))

mse_fold <- rep(0, K)
for (j in 1:K) {
  train_j <- datos[folds != j, ]
  test_j  <- datos[folds == j, ]

  modelo <- lm(y ~ ., data = train_j)
  pred <- predict(modelo, test_j)
  mse_fold[j] <- mean((test_j$y - pred)^2)
}

mean(mse_fold)   # estimador CV
sd(mse_fold)     # variabilidad entre folds


# cv.glm del paquete boot (para modelos glm o lm)
library(boot)
modelo_glm <- glm(Grad.Rate ~ ., data = College[, -1])

cv10 <- cv.glm(College[, -1], modelo_glm, K = 10)
cv10$delta[1]    # estimador CV crudo
cv10$delta[2]    # estimador ajustado por sesgo

# LOOCV: omitir el argumento K
cv_loo <- cv.glm(College[, -1], modelo_glm)
cv_loo$delta[1]


# Con el paquete caret: calibrar k vecinos por CV
library(caret)

control <- trainControl(method = "cv", number = 10)
ajuste <- train(Grad.Rate ~ ., data = College[, -1],
                method = "knn",
                trControl = control,
                tuneGrid = data.frame(k = 1:30),
                preProcess = c("center", "scale"))

ajuste
plot(ajuste)
ajuste$bestTune
