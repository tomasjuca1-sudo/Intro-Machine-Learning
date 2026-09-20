# ==========================================================================
# TEMA 04: K-MEDOIDES
# Clase 3
# ==========================================================================


# --------------------------------------------------------------------------
# K-Medoides con distancia euclidiana
# Origen: Del curso, class3_book1_kmedoids.ipynb
# --------------------------------------------------------------------------

from scipy.spatial import distance_matrix, distance
import numpy as np
import matplotlib.pyplot as plt

# La entrada del algoritmo es la matriz de distancias n x n
dista_s = distance_matrix(Xs, Xs)
np.shape(dista_s)

#%pip install kmedoids   # quitar el # si falta instalar
import kmedoids

c = kmedoids.fasterpam(dista_s, 3)
print("Loss is:", c.loss)

labels = c.labels      # etiquetas de grupo
medoids = c.medoids    # ÍNDICES de las observaciones medoide

plt.scatter(Xs[:, 2], Xs[:, 3], c=labels, cmap='viridis', alpha=0.6)
plt.title("K-Medoids Clustering")
plt.show()


# --------------------------------------------------------------------------
# K-Medoides con distancia de Gower (datos no euclidianos)
# Origen: Del curso, class3_book1_kmedoids.ipynb
# --------------------------------------------------------------------------

#!pip install gower   # quitar el # si falta instalar
import gower

from sklearn.datasets import load_wine
wine = load_wine()
W = wine.data

# Matriz de distancias tipo Gower: sirve para variables mixtas o categóricas
dist_g = gower.gower_matrix(W)
np.shape(dist_g)

med_w = kmedoids.fasterpam(dist_g, 3)

labels = med_w.labels
medoids = med_w.medoids

print(medoids)   # índices de los medoides
print(labels)


# --------------------------------------------------------------------------
# Alternativa con sklearn_extra
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

# Si kmedoids no está disponible
from sklearn_extra.cluster import KMedoids

modelo = KMedoids(n_clusters=3, metric='euclidean', method='pam', random_state=0)
etiquetas = modelo.fit_predict(Xs)

modelo.medoid_indices_    # índices de los medoides
modelo.cluster_centers_   # coordenadas de los medoides
modelo.inertia_           # suma de distancias a los medoides

# Con una matriz de distancias precalculada
modelo_pre = KMedoids(n_clusters=3, metric='precomputed', method='pam', random_state=0)
etiquetas_pre = modelo_pre.fit_predict(dista_s)
