# ==========================================================================
# TEMA 15: EPE, MSE Y EL BALANCE SESGO VARIANZA
# Clase 10, Clase 11
# ==========================================================================


# --------------------------------------------------------------------------
# Curva en U en R con k vecinos
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

library(FNN)

# Escenario controlado
set.seed(1)
a <- 0.3
b <- 0.2
f <- function(x) 2 + x^a * cos(x/b)

n <- 400
x <- runif(n, 1, 5)
sd_err <- 1.2
y <- f(x) + rnorm(n, 0, sd_err)

# Partición
prop_train <- 0.75
n_train <- floor(n*prop_train)
idx <- sample(1:n, n_train)

x_train <- x[idx] ; y_train <- y[idx]
x_test  <- x[-idx] ; y_test  <- y[-idx]

# Curva de MSE variando k
ks <- 1:60
mse_train <- rep(0, length(ks))
mse_test  <- rep(0, length(ks))

for (i in seq_along(ks)) {
  k <- ks[i]
  pred_tr <- knn.reg(train = matrix(x_train), test = matrix(x_train),
                     y = y_train, k = k)$pred
  pred_te <- knn.reg(train = matrix(x_train), test = matrix(x_test),
                     y = y_train, k = k)$pred
  mse_train[i] <- mean((pred_tr - y_train)^2)
  mse_test[i]  <- mean((pred_te - y_test)^2)
}

plot(ks, mse_test, type = "l", col = "blue", lwd = 3,
     ylim = range(c(mse_train, mse_test)),
     xlab = "k (k grande = menos flexible)", ylab = "MSE")
lines(ks, mse_train, col = "red", lwd = 3)
abline(h = sd_err^2, lty = 2)   # error irreducible

k_opt <- ks[which.min(mse_test)]
k_opt
min(mse_test)
