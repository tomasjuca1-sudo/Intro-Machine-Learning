# ==========================================================================
# TEMA 01: AXIOMA DE LOS DATOS Y TIPOS DE APRENDIZAJE
# Clase 1, Clase 10
# ==========================================================================


# --------------------------------------------------------------------------
# Simulación supervisada: f_optima más ruido
# Origen: Del curso, class10_book1_intro.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Parámetros para generar la función de media (los puede cambiar)
a = 0.3
b = 0.2

# Definición de la función f_optima
def f(x):
    z = 2 + x**(a) * math.cos(x/b)
    return(z)

# Visualización de la función f_optima
points = np.arange(1.0, 5.0, 0.01)
f_mean = np.frompyfunc(f, 1, 1)
plt.plot(points, f_mean(points))

# Generación de datos de entrenamiento
n = 400                              # Tamaño de muestra
x = np.random.uniform(1.0, 5.0, n)   # Generación de X
sd = 1.2                             # Desviación estándar del error irreducible
y = f_mean(x) + np.random.normal(0, sd, n)

plt.plot(x, y, 'o')


# --------------------------------------------------------------------------
# Efecto del tamaño de muestra en la varianza
# Origen: Del curso, class10_book1_intro.ipynb
# --------------------------------------------------------------------------

n = 10   # Muestra pequeña

x1 = np.random.uniform(1.0, 5.0, n)
x2 = np.random.uniform(1.0, 5.0, n)

sd = 1.2
y1 = f_mean(x1) + np.random.normal(0, sd, n)
y2 = f_mean(x2) + np.random.normal(0, sd, n)

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].plot(x1, y1, 'o')
axs[0].set_title('Muestra 1')
axs[1].plot(x2, y2, 'o', color="red")
axs[1].set_title('Muestra 2')
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------
# Simulación no supervisada: normal multivariada y mezcla por grupos
# Origen: Del curso, class1_book1_intro.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random

# Vector de medias y matriz de covarianza
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.5],
                [0.5, 1.0]])

rng = np.random.default_rng(seed=42)
n = 100

data = rng.multivariate_normal(mean, cov, size=n)
plt.scatter(data[:, 0], data[:, 1], color='blue', marker='o')

# Ahora los datos forman grupos (mezcla de densidades)
p1 = 0.6
p2 = 0.4

m1 = np.array([0.0, 1.0])
m2 = np.array([4.5, 6.0])

data2 = np.zeros((n, 2))

for i in range(n):
    u = random.random()
    if u < p1:
        data2[i, :] = rng.multivariate_normal(m1, cov, size=1)
    else:
        data2[i, :] = rng.multivariate_normal(m2, cov, size=1)

plt.scatter(data2[:, 0], data2[:, 1], color='blue', marker='o')
