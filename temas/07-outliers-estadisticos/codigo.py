# ==========================================================================
# TEMA 07: OUTLIERS: Z-SCORE Y MAHALANOBIS
# Clase 5
# ==========================================================================


# --------------------------------------------------------------------------
# Z-Score sobre una muestra contaminada
# Origen: Del curso, class5_book1_outlier.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng()

mean = 5.0
std_dev = 1.5
n = 80

samples = rng.normal(loc=mean, scale=std_dev, size=n)

count, bins, ignored = plt.hist(samples, 10, density=True)
plt.plot(bins, 1/(std_dev * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mean)**2 / (2 * std_dev**2)),
         linewidth=2, color='r')
plt.show()

# Se contamina la muestra con un valor extremo
samples[n-1] = 12.3

count, bins, ignored = plt.hist(samples, 10, density=True)
plt.plot(bins, 1/(std_dev * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mean)**2 / (2 * std_dev**2)),
         linewidth=2, color='r')
plt.show()

# Z-Score
z_scores = stats.zscore(samples)
print(z_scores)

plt.plot(z_scores, linewidth=2, color='r')

# Regla práctica
print("Outliers con |Z| > 3:", np.where(np.abs(z_scores) > 3)[0])


# --------------------------------------------------------------------------
# Distancia de Mahalanobis
# Origen: Del curso, class5_book1_outlier.ipynb
# --------------------------------------------------------------------------

from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

mean = [2, 3]
cov = [[3, 1.5], [1.5, 4]]
n = 200

x = np.random.multivariate_normal(mean, cov, n).T

plt.plot(x[0, :], x[1, :], 'x')
plt.axis('equal')
plt.show()

mean2 = np.mean(x, axis=0)

cov_matrix = np.cov(x.T, rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)

distances = [distance.mahalanobis(x, mean2, inv_cov) for x in x]

# Versión matricial: la respuesta está en la diagonal
left = np.dot((x-mean2).T, inv_cov)
mahal = np.dot(left, (x-mean2))
mahal.diagonal()


# --------------------------------------------------------------------------
# Mahalanobis con umbral chi cuadrado (versión ordenada)
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

import numpy as np
from scipy.stats import chi2

# Datos con forma n x p (cada FILA una observación)
n = 200
p = 2
X = np.random.multivariate_normal([2, 3], [[3, 1.5], [1.5, 4]], n)

media = np.mean(X, axis=0)
S = np.cov(X, rowvar=False)
S_inv = np.linalg.inv(S)

centrado = X - media
d2 = np.sum(centrado @ S_inv * centrado, axis=1)   # d_M^2 de cada punto

# Umbral: cuantil 0.975 de una chi cuadrado con p grados de libertad
umbral = chi2.ppf(0.975, df=p)
outliers = np.where(d2 > umbral)[0]

print("Umbral chi2:", round(umbral, 3))
print("Índices atípicos:", outliers)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=(d2 > umbral), cmap='coolwarm')
plt.title("Outliers por distancia de Mahalanobis")
plt.show()
