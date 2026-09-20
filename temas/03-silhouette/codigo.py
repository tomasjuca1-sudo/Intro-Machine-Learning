# ==========================================================================
# TEMA 03: SILHOUETTE Y MÉTRICAS DE AJUSTE
# Clase 3, Clase 4
# ==========================================================================


# --------------------------------------------------------------------------
# Las tres métricas en una línea cada una
# Origen: Del curso, class2_book1_kmeans.ipynb
# --------------------------------------------------------------------------

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

sil = silhouette_score(Xs, y_kmeans_s)
ch = calinski_harabasz_score(Xs, y_kmeans_s)
db = davies_bouldin_score(Xs, y_kmeans_s)

print(f"Clusters: {K} | Silhouette: {sil:.3f} | Calinski-Harabasz: {ch:.1f} | Davies-Bouldin: {db:.3f}")


# --------------------------------------------------------------------------
# Gráfico de Silhouette con yellowbrick
# Origen: Del curso, class2_book1_kmeans.ipynb
# --------------------------------------------------------------------------

from sklearn.datasets import make_blobs
from yellowbrick.cluster import SilhouetteVisualizer

visualizer = SilhouetteVisualizer(kmeans_s, colors='yellowbrick')
visualizer.fit(Xs)
visualizer.show()


# --------------------------------------------------------------------------
# Escoger K comparando Silhouette (sin yellowbrick)
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt

resultados = []
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=0)
    etiquetas = modelo.fit_predict(Xs)
    resultados.append((k, silhouette_score(Xs, etiquetas)))
    print(k, round(silhouette_score(Xs, etiquetas), 3))

ks = [r[0] for r in resultados]
sils = [r[1] for r in resultados]
plt.plot(ks, sils, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette promedio")
plt.show()

# Silhouette individual de cada observación (los s(i))
modelo = KMeans(n_clusters=3, n_init=10, random_state=0)
etiquetas = modelo.fit_predict(Xs)
s_i = silhouette_samples(Xs, etiquetas)

# Cuántos puntos quedaron mal asignados (s(i) negativo)
print("Puntos con s(i) < 0:", np.sum(s_i < 0))
