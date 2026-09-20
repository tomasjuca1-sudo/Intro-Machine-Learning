# ==========================================================================
# TEMA 12: KERNEL PCA
# Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# KernelPCA con sklearn
# Origen: Equivalente agregado. El código del curso está en R
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.datasets import make_circles

# Datos no separables linealmente
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Datos originales")
plt.show()

# PCA lineal: no separa
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(X)
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y)
plt.title("PCA lineal")
plt.show()

# Kernel PCA con kernel gaussiano
# gamma de sklearn cumple el papel de sigma de kernlab
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10, random_state=0)
Z_kpca = kpca.fit_transform(X)

plt.scatter(Z_kpca[:, 0], Z_kpca[:, 1], c=y)
plt.title("Kernel PCA (rbf)")
plt.show()

# Proyectar datos nuevos
X_nuevo, _ = make_circles(n_samples=50, factor=0.3, noise=0.05, random_state=1)
Z_nuevo = kpca.transform(X_nuevo)


# --------------------------------------------------------------------------
# Comparar kernels y calibrar gamma
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

kernels = ['linear', 'poly', 'rbf', 'cosine']

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, k in zip(axs, kernels):
    modelo = KernelPCA(n_components=2, kernel=k, gamma=10, degree=3, random_state=0)
    Z = modelo.fit_transform(X)
    ax.scatter(Z[:, 0], Z[:, 1], c=y, s=10)
    ax.set_title(k)
plt.tight_layout()
plt.show()

# Efecto de gamma en el kernel rbf
for g in [0.1, 1, 10, 100]:
    modelo = KernelPCA(n_components=2, kernel='rbf', gamma=g, random_state=0)
    Z = modelo.fit_transform(X)
    plt.figure(figsize=(3, 3))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8)
    plt.title(f"gamma = {g}")
    plt.show()

# Con reconstrucción aproximada (pre-image)
kpca_inv = KernelPCA(n_components=2, kernel='rbf', gamma=10,
                     fit_inverse_transform=True, alpha=0.1, random_state=0)
Z = kpca_inv.fit_transform(X)
X_recon = kpca_inv.inverse_transform(Z)
