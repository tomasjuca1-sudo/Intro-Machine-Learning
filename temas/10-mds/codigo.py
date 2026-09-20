# ==========================================================================
# TEMA 10: MULTIDIMENSIONAL SCALING (MDS)
# Clase 7, Clase 8
# ==========================================================================


# --------------------------------------------------------------------------
# MDS clásico, métrico y no métrico con sklearn
# Origen: Equivalente agregado. El código de MDS del curso está en R
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Con datos: primero se calculan las distancias
from sklearn.datasets import load_iris
X = load_iris().data
D = pairwise_distances(X, metric='euclidean')

# MDS métrico (usa los valores de las distancias)
mds_metrico = MDS(n_components=2, dissimilarity='precomputed',
                  normalized_stress='auto', random_state=0)
Z = mds_metrico.fit_transform(D)

print("Stress:", mds_metrico.stress_)

plt.scatter(Z[:, 0], Z[:, 1], c=load_iris().target)
plt.title("MDS métrico")
plt.show()

# MDS no métrico (solo usa el orden de las distancias)
mds_no_metrico = MDS(n_components=2, dissimilarity='precomputed',
                     metric=False, normalized_stress='auto', random_state=0)
Z_nm = mds_no_metrico.fit_transform(D)

plt.scatter(Z_nm[:, 0], Z_nm[:, 1], c=load_iris().target)
plt.title("MDS no métrico")
plt.show()


# --------------------------------------------------------------------------
# MDS clásico a mano (doble centrado)
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

import numpy as np

def mds_clasico(D, d=2):
    n = D.shape[0]

    # Doble centrado: B = -1/2 * J D^2 J
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ (D**2) @ J

    # Descomposición espectral de B
    valores, vectores = np.linalg.eigh(B)
    orden = np.argsort(valores)[::-1]
    valores = valores[orden]
    vectores = vectores[:, orden]

    # Coordenadas: Z = V_d * sqrt(Lambda_d)
    L = np.diag(np.sqrt(np.maximum(valores[:d], 0)))
    Z = vectores[:, :d] @ L

    return Z, valores

Z, eigenvalores = mds_clasico(D, d=2)

print("Primeros eigenvalores:", eigenvalores[:5])
print("Proporción explicada:", np.sum(np.abs(eigenvalores[:2]))/np.sum(np.abs(eigenvalores)))


# --------------------------------------------------------------------------
# Distancia de Gower en Python para MDS con datos mixtos
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

#!pip install gower   # quitar el # si falta instalar
import gower
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# W puede tener variables continuas y categóricas mezcladas
D_gower = gower.gower_matrix(W)

mds = MDS(n_components=2, dissimilarity='precomputed',
          normalized_stress='auto', random_state=0)
Z = mds.fit_transform(D_gower)

plt.scatter(Z[:, 0], Z[:, 1])
plt.title("MDS sobre distancia de Gower")
plt.show()
