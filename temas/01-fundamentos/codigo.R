# ==========================================================================
# TEMA 01: AXIOMA DE LOS DATOS Y TIPOS DE APRENDIZAJE
# Clase 1, Clase 10
# ==========================================================================


# --------------------------------------------------------------------------
# Equivalente en R de la simulación
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Simulación supervisada
a <- 0.3
b <- 0.2
f <- function(x) 2 + x^a * cos(x/b)

n <- 400
x <- runif(n, 1, 5)
sd_err <- 1.2
y <- f(x) + rnorm(n, 0, sd_err)

plot(x, y, pch = 16, cex = 0.6)
curve(f, from = 1, to = 5, col = "red", lwd = 2, add = TRUE)


# Simulación no supervisada: mezcla de dos normales bivariadas
library(MASS)

mu1 <- c(0, 1)
mu2 <- c(4.5, 6)
Sigma <- matrix(c(1, 0.5, 0.5, 1), nrow = 2)

n <- 100
p1 <- 0.6
grupo <- ifelse(runif(n) < p1, 1, 2)

datos <- matrix(0, nrow = n, ncol = 2)
for (i in 1:n) {
  if (grupo[i] == 1) {
    datos[i, ] <- mvrnorm(1, mu1, Sigma)
  } else {
    datos[i, ] <- mvrnorm(1, mu2, Sigma)
  }
}

plot(datos, col = grupo, pch = 16)
