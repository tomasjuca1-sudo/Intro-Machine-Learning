# ==========================================================================
# TEMA 02: K-MEANS
# Clase 2, Clase 3
# ==========================================================================


# --------------------------------------------------------------------------
# K-Means paso a paso (implementación manual)
# Origen: Del curso, class2_book1_kmeans.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multinomial
from sklearn import datasets
from sklearn.datasets import load_iris

# Datos a trabajar: IRIS
iris = load_iris()
X = iris.data

n = np.shape(X)[0]   # tamaño de muestra
p = np.shape(X)[1]   # número de variables
K = 3                # número de clusters

# Asignación aleatoria inicial de cada observación a un grupo
probabilities = [1/K]*K
assign_matrix = multinomial.rvs(n=1, p=probabilities, size=n, random_state=42)

# Tamaños de grupo y matriz de medias inicial (p x K)
n_group = sum(assign_matrix)
means = np.matmul(X.T, assign_matrix)/n_group

# Matriz de distancias de cada punto a cada centro (n x K)
dista = np.zeros((n, K))
for i in range(K):
    dista[:, i] = np.sum((X-means[:, i])*(X-means[:, i]), axis=1)

# Loop principal del algoritmo
tol = 0.00001
sse_hist = [np.sum(np.diag(np.matmul(dista.T, assign_matrix)))]
change = 10000

while change > tol:
    n_group = sum(assign_matrix)
    means = np.matmul(X.T, assign_matrix)/n_group

    dista = np.zeros((n, K))
    for i in range(K):
        dista[:, i] = np.sum((X-means[:, i])*(X-means[:, i]), axis=1)

    assign_matrix = np.zeros((n, K))
    for i in range(n):
        assign_matrix[i, np.argmin(dista[i, :])] = 1

    sse = np.sum(np.diag(np.matmul(dista.T, assign_matrix)))
    sse_hist.append(sse)
    change = sse_hist[-2] - sse

# Convergencia del SSE
plt.plot(sse_hist)
plt.xlabel("Iteración")
plt.ylabel("SSE")

# Grupos encontrados
col = np.matmul(assign_matrix, [1, 5, 8])
plt.scatter(X[:, 2], X[:, 3], c=col, alpha=0.7)


# --------------------------------------------------------------------------
# K-Means con scikit-learn y escalamiento
# Origen: Del curso, class2_book1_kmeans.ipynb
# --------------------------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# Sin escalar
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(X)

assign = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_   # cada FILA es la media de un grupo
y_kmeans = kmeans.labels_

plt.scatter(X[:, 2], X[:, 3], c=y_kmeans, alpha=0.7)

# Escalando las variables (importante si están en unidades distintas)
Xs = scale(datasets.load_iris().data)

kmeans_s = KMeans(n_clusters=K, random_state=0)
kmeans_s.fit(Xs)

assign_s = kmeans_s.fit_predict(Xs)
centroids_s = kmeans_s.cluster_centers_
y_kmeans_s = kmeans_s.labels_

plt.scatter(Xs[:, 2], Xs[:, 3], c=y_kmeans_s, alpha=0.7)


# --------------------------------------------------------------------------
# Método del codo para escoger K
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sse_por_k = []
rango_k = range(1, 11)

for k in rango_k:
    modelo = KMeans(n_clusters=k, n_init=10, random_state=0)
    modelo.fit(Xs)
    sse_por_k.append(modelo.inertia_)   # inertia_ es el WSS

plt.plot(list(rango_k), sse_por_k, marker='o')
plt.xlabel("Número de clusters K")
plt.ylabel("SSE (within)")
plt.title("Método del codo")
plt.show()
