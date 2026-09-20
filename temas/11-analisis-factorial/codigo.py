# ==========================================================================
# TEMA 11: ANÁLISIS FACTORIAL
# Clase 7, Clase 8, Clase 9
# ==========================================================================


# --------------------------------------------------------------------------
# Análisis factorial con factor_analyzer
# Origen: Equivalente agregado. El código del curso está en R
# --------------------------------------------------------------------------

#!pip install factor_analyzer   # quitar el # si falta instalar
from factor_analyzer import FactorAnalyzer
import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data

# Un factor, rotación varimax
fa = FactorAnalyzer(n_factors=1, rotation='varimax', method='ml')
fa.fit(X)

fa.loadings_              # cargas (Lambda), p x d
fa.get_uniquenesses()     # unicidades
fa.get_communalities()    # comunalidades
fa.get_factor_variance()  # varianza explicada por factor

# Scores de cada observación
scores = fa.transform(X)

# Rotación oblicua
fa_promax = FactorAnalyzer(n_factors=3, rotation='promax', method='ml')

# Decidir el número de factores: gráfico de eigenvalores
fa_todos = FactorAnalyzer(n_factors=X.shape[1], rotation=None, method='ml')
fa_todos.fit(X)
ev, v = fa_todos.get_eigenvalues()

import matplotlib.pyplot as plt
plt.plot(range(1, len(ev)+1), ev, marker='o')
plt.axhline(1, color='red', linestyle='--')
plt.xlabel("Factor")
plt.ylabel("Eigenvalor")
plt.show()


# --------------------------------------------------------------------------
# FactorAnalysis de sklearn
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.decomposition import FactorAnalysis
import numpy as np

fa = FactorAnalysis(n_components=1, random_state=0)
scores = fa.fit_transform(X)

fa.components_    # cargas, una FILA por factor
fa.noise_variance_   # unicidades (Psi)

# Comunalidades
comunalidades = np.sum(fa.components_**2, axis=0)
print(comunalidades)

# Visualizar la densidad del factor por especie de iris
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

y = load_iris().target
for especie in [0, 1, 2]:
    s = scores[y == especie, 0]
    kde = gaussian_kde(s)
    rejilla = np.linspace(scores.min(), scores.max(), 200)
    plt.plot(rejilla, kde(rejilla), lw=2)
plt.title("Densidad del factor por especie")
plt.show()
