# ==========================================================================
# TEMA 06: DBSCAN
# Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# Calibración de eps y ajuste de DBSCAN
# Origen: Del curso, class4_book2_DBSCAN.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Función para escanear la distancia al k-ésimo vecino
def distancia_kveci(X, k=3):
    kveci = NearestNeighbors(n_neighbors=k)
    veci = kveci.fit(X)
    distances, indices = veci.kneighbors(X)
    return distances[:, 1:].reshape(-1)   # se quita la columna 0 (el punto mismo)

iris = load_iris()
X = iris.data

distancia_kveci(X, 2)

# Distancias ordenadas: el codo sugiere el radio eps
plt.plot(sorted(distancia_kveci(X, 5)))

# Ajuste del modelo
dbsca = DBSCAN(eps=0.6, min_samples=6).fit(X)

label = dbsca.labels_   # la clase -1 corresponde a los puntos de ruido (noise)
label

plt.scatter(X[:, 2], X[:, 3], c=label)
plt.show()


# --------------------------------------------------------------------------
# Datos con formas irregulares (círculos concentricos más ruido)
# Origen: Del curso, class4_book2_DBSCAN.ipynb
# --------------------------------------------------------------------------

np.random.seed(42)

# Círculos concentricos
theta = np.linspace(0, 2 * np.pi, 300)
r1 = 2 + np.random.normal(0, 0.4, 300)
circle1_x, circle1_y = r1 * np.cos(theta), r1 * np.sin(theta)

r2 = 5 + np.random.normal(0, 0.4, 300)
circle2_x, circle2_y = r2 * np.cos(theta), r2 * np.sin(theta)

# Ruido / outliers
noise_x = np.random.uniform(-8, 8, 100)
noise_y = np.random.uniform(-8, 8, 100)

X = np.vstack(
    [
        np.column_stack((circle1_x, circle1_y)),
        np.column_stack((circle2_x, circle2_y)),
        np.column_stack((noise_x, noise_y)),
    ]
)

plt.scatter(X[:, 0], X[:, 1], s=10, c="purple")
plt.title("Synthetic Multishapes Dataset")
plt.show()

# Calibrar y ajustar
plt.plot(sorted(distancia_kveci(X, 8)))

dbsca = DBSCAN(eps=0.7, min_samples=8).fit(X)
label = dbsca.labels_
print(label)

plt.scatter(X[:, 0], X[:, 1], c=label)
plt.show()


# --------------------------------------------------------------------------
# Resumen de la solución: número de clusters y de outliers
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

import numpy as np

label = dbsca.labels_

n_clusters = len(set(label)) - (1 if -1 in label else 0)
n_ruido = list(label).count(-1)

print("Número de clusters:", n_clusters)
print("Número de puntos de ruido:", n_ruido)
print("Proporción de ruido:", round(n_ruido/len(label), 3))

# Tamaño de cada grupo
valores, conteos = np.unique(label, return_counts=True)
for v, c in zip(valores, conteos):
    nombre = "ruido" if v == -1 else f"cluster {v}"
    print(nombre, c)
