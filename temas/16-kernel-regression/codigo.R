# ==========================================================================
# TEMA 16: KERNEL REGRESSION Y K VECINOS
# Clase 10, Clase 11
# ==========================================================================


# --------------------------------------------------------------------------
# Kernel regression y k vecinos en R
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Implementación manual del kernel rectangular
kernelreg <- function(xstar, X, Y, h) {
  pred <- rep(0, length(xstar))
  for (i in seq_along(xstar)) {
    dentro <- abs(xstar[i] - X) <= h
    pred[i] <- sum(Y[dentro]) / max(sum(dentro), 0.1)
  }
  return(pred)
}

xstar <- seq(1, 5, by = 0.01)

pred_ancho <- kernelreg(xstar, x_train, y_train, 0.4)    # sesgo
pred_angosto <- kernelreg(xstar, x_train, y_train, 0.03) # varianza

plot(x_train, y_train, pch = 16, cex = 0.4)
curve(f, from = 1, to = 5, col = "red", lwd = 3, add = TRUE)
lines(xstar, pred_ancho, col = "blue", lwd = 2)


# Calibración de h
hh <- seq(0.01, 1, by = 0.002)
mse_train <- rep(0, length(hh))
mse_test <- rep(0, length(hh))

for (i in seq_along(hh)) {
  fhat_train <- kernelreg(x_train, x_train, y_train, hh[i])
  fhat_test  <- kernelreg(x_test,  x_train, y_train, hh[i])
  mse_train[i] <- mean((fhat_train - y_train)^2)
  mse_test[i]  <- mean((fhat_test - y_test)^2)
}

plot(hh, mse_train, type = "l", col = "red", lwd = 3,
     ylim = range(c(mse_train, mse_test)))
lines(hh, mse_test, col = "blue", lwd = 3)

h_opt <- hh[which.min(mse_test)]
h_opt


# Funciones ya implementadas en R
# ksmooth: kernel regression con ventana box o normal
ks <- ksmooth(x_train, y_train, kernel = "box", bandwidth = 2*h_opt, x.points = xstar)
lines(ks$x, ks$y, col = "darkgreen", lwd = 2)

# loess: regresión local polinomial
lo <- loess(y_train ~ x_train, span = 0.2)
lines(xstar, predict(lo, data.frame(x_train = xstar)), col = "purple", lwd = 2)

# k vecinos
library(FNN)
pred_knn <- knn.reg(train = matrix(x_train), test = matrix(xstar),
                    y = y_train, k = 10)$pred
