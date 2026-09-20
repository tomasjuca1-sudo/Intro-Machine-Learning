# ==========================================================================
# TEMA 15: EPE, MSE Y EL BALANCE SESGO VARIANZA
# Clase 10, Clase 11
# ==========================================================================


# --------------------------------------------------------------------------
# Escenario controlado: f_optima, sigma y muestras
# Origen: Del curso, class10_book1_intro.ipynb
# --------------------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt
import random

a = 0.3
b = 0.2

def f(x):
    z = 2 + x**(a) * math.cos(x/b)
    return(z)

points = np.arange(1.0, 5.0, 0.01)
f_mean = np.frompyfunc(f, 1, 1)
plt.plot(points, f_mean(points))

n = 400
x = np.random.uniform(1.0, 5.0, n)
sd = 1.2                              # sigma: error irreducible
y = f_mean(x) + np.random.normal(0, sd, n)

plt.plot(x, y, 'o')


# --------------------------------------------------------------------------
# Partición train y test manual
# Origen: Del curso, class10_book2_kernel.ipynb
# --------------------------------------------------------------------------

prop_train = 0.75
n_train = int(n*prop_train)
train_index = random.sample(range(len(x)), n_train)

x_train = x[train_index]
y_train = y[train_index]

x_test = np.delete(x, train_index)
y_test = np.delete(y, train_index)

# Sin semilla la partición cambia en cada corrida.
# La alternativa es train_test_split de sklearn.model_selection


# --------------------------------------------------------------------------
# Curva en U: MSE de train contra MSE de test
# Origen: Del curso, class10_book2_kernel.ipynb
# --------------------------------------------------------------------------

hh = np.arange(0.01, 1, 0.002)   # secuencia de niveles de flexibilidad

mse_train = np.repeat(0.0, len(hh))
mse_test = np.repeat(0.0, len(hh))

for i in range(len(hh)):
    fhat_train = kernelreg(x_train, x_train, y_train, hh[i])
    fhat_test = kernelreg(x_test, x_train, y_train, hh[i])
    mse_train[i] = (sum((fhat_train-y_train)**2)/n_train)
    mse_test[i] = sum((fhat_test-y_test)**2)/(n-n_train)

# Rojo: MSE de train. Cuando h va a cero, el MSE va a cero
# Azul: MSE de test. Función convexa
plt.plot(hh, mse_train, "r", linewidth='3')
plt.plot(hh, mse_test, "b", linewidth='3')

# Punto calibrado
h_opt = hh[np.argmin(mse_test)]
mse_opt = min(mse_test)

print("h óptimo:", h_opt)
print("MSE mínimo en test:", mse_opt)
