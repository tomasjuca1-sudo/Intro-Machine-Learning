# ==========================================================================
# TEMA 13: SVD (DESCOMPOSICIÓN EN VALORES SINGULARES)
# Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# SVD y aproximación de rango bajo con numpy
# Origen: Equivalente agregado. El código del curso está en R
# --------------------------------------------------------------------------

import numpy as np

# X es la matriz a descomponer
U, d, Vt = np.linalg.svd(X, full_matrices=False)

print("Valores singulares:", d)

# Verificación: reconstruye la matriz original
reconstruida = U @ np.diag(d) @ Vt
print("Error de reconstrucción completo:", np.max(np.abs(X - reconstruida)))

# Aproximación de rango r
r = 10
X_r = U[:, :r] @ np.diag(d[:r]) @ Vt[:r, :]

print("Error con rango", r, ":", np.linalg.norm(X - X_r, 'fro'))

# Proporción de información retenida
print("Retenido:", np.sum(d[:r]**2)/np.sum(d**2))

# Gráfico de valores singulares (decide r)
import matplotlib.pyplot as plt
plt.plot(d, marker='o')
plt.xlabel("Índice")
plt.ylabel("Valor singular")
plt.show()


# --------------------------------------------------------------------------
# Matriz término documento, LSI y búsqueda por coseno
# Origen: Equivalente agregado. Replica el ejemplo de information retrieval del curso
# --------------------------------------------------------------------------

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

documentos = [
    "oil prices rise in the crude market",
    "crude oil production falls sharply",
    "stock market rallies after the report",
    "prices of oil and gas increase",
]

# Matriz documento término (sklearn la arma al revés que R)
vectorizador = CountVectorizer(stop_words='english')
DT = vectorizador.fit_transform(documentos).toarray()

print(vectorizador.get_feature_names_out())
print(DT.shape)

# SVD truncado: Latent Semantic Indexing
r = 2
svd = TruncatedSVD(n_components=r, random_state=0)
documentos_reducidos = svd.fit_transform(DT)

print("Varianza explicada:", svd.explained_variance_ratio_.sum())

# Consulta: se proyecta al mismo espacio
consulta = vectorizador.transform(["crude oil prices"]).toarray()
consulta_reducida = svd.transform(consulta)

# Similitud de coseno contra cada documento
similitudes = cosine_similarity(consulta_reducida, documentos_reducidos)[0]

for i, s in enumerate(similitudes):
    print(f"Documento {i}: coseno = {s:.3f}")

ranking = np.argsort(similitudes)[::-1]
print("Ranking de relevancia:", ranking)
