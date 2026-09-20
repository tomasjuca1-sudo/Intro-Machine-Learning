# ==========================================================================
# TEMA 09: PCA (COMPONENTES PRINCIPALES)
# Clase 6, Clase 7, Clase 8
# ==========================================================================


# --------------------------------------------------------------------------
# PCA con sklearn (equivalente del código en R)
# Origen: Equivalente agregado. El código de PCA del curso está en R
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Escalar si las variables tienen unidades distintas
Xs = StandardScaler().fit_transform(X)

pca = PCA()            # sin argumento calcula todos los componentes
scores = pca.fit_transform(Xs)

pca.explained_variance_          # eigenvalores (varianza de cada componente)
pca.explained_variance_ratio_    # proporción de varianza explicada
np.cumsum(pca.explained_variance_ratio_)   # varianza acumulada
pca.components_                  # cargas (loadings), una FILA por componente

# Gráfico de codo
plt.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
plt.xlabel("Componente")
plt.ylabel("Varianza (eigenvalor)")
plt.title("Varianzas de Componentes")
plt.show()

# Proyección en los dos primeros componentes
plt.scatter(scores[:, 0], scores[:, 1], c=iris.target)
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()


# --------------------------------------------------------------------------
# PCA a mano con descomposición espectral y reconstrucción
# Origen: Equivalente agregado. Replica el procedimiento manual del código en R
# --------------------------------------------------------------------------

import numpy as np

# Matriz de covarianza
S = np.cov(X, rowvar=False)

# Descomposición espectral
valores, vectores = np.linalg.eigh(S)

# eigh devuelve en orden ascendente: se invierte
orden = np.argsort(valores)[::-1]
valores = valores[orden]
vectores = vectores[:, orden]

print("Eigenvalores:", valores)
print("Proporción explicada:", valores/np.sum(valores))

# Datos centrados y scores
medias = X.mean(axis=0)
Xc = X - medias
Z = Xc @ vectores

# Reconstrucción con k componentes
k = 2
X_aprox = Z[:, :k] @ vectores[:, :k].T + medias

error = np.mean((X - X_aprox)**2)
print("Error cuadrático medio de reconstrucción con", k, "componentes:", round(error, 4))


# --------------------------------------------------------------------------
# Biplot en Python
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

def biplot(scores, cargas, nombres):
    xs = scores[:, 0]
    ys = scores[:, 1]
    escala_x = 1/(xs.max() - xs.min())
    escala_y = 1/(ys.max() - ys.min())

    plt.scatter(xs*escala_x, ys*escala_y, s=10, alpha=0.5)
    for i, nombre in enumerate(nombres):
        plt.arrow(0, 0, cargas[0, i], cargas[1, i], color='r', alpha=0.7)
        plt.text(cargas[0, i]*1.1, cargas[1, i]*1.1, nombre, color='r')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.show()

biplot(scores, pca.components_, iris.feature_names)
