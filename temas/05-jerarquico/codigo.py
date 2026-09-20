# ==========================================================================
# TEMA 05: CLUSTERING JERÁRQUICO
# Clase 3, Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# Aglomerativo y dendrograma
# Origen: Del curso, class4_book1_hierarquical.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
n = np.shape(X)[0]
p = np.shape(X)[1]
K = 3

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Etiquetas: linkage puede ser 'single', 'complete', 'average' o 'ward'
hierarchical_cluster = AgglomerativeClustering(n_clusters=3, linkage='single')
labels = hierarchical_cluster.fit_predict(X)

plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.show()

# Dendrograma
linkage_data = linkage(X, method='single', metric='euclidean')
dendrogram(linkage_data)
plt.show()


# --------------------------------------------------------------------------
# Comparar los cuatro linkages y cortar el árbol
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

Xs = scale(X)

metodos = ['single', 'complete', 'average', 'ward']

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, metodo in zip(axs.flat, metodos):
    arbol = linkage(Xs, method=metodo, metric='euclidean')
    dendrogram(arbol, ax=ax, no_labels=True)
    ax.set_title(metodo)
plt.tight_layout()
plt.show()

# Cortar el dendrograma para obtener K grupos
arbol = linkage(Xs, method='ward', metric='euclidean')

etiquetas_k = fcluster(arbol, t=3, criterion='maxclust')     # cortar en 3 grupos
etiquetas_h = fcluster(arbol, t=10, criterion='distance')    # cortar a altura 10

print(etiquetas_k)
