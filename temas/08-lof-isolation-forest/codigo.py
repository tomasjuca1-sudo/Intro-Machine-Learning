# ==========================================================================
# TEMA 08: LOF E ISOLATION FOREST
# Clase 6
# ==========================================================================


# --------------------------------------------------------------------------
# Datos simulados con tres grupos más outliers
# Origen: Del curso, class6_book1_outlier_lof.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy.linalg import sqrtm

vcov1 = np.array([[2.0, 1.5], [1.5, 4.0]])
vcov2 = np.array([[2.0, -1.5], [-1.5, 4.0]])

m1 = [-4, -2]
m2 = [1, -5]
m3 = [4, 3]

n1 = 50 ; n2 = 80 ; n3 = 60

X1 = m1 + 0.5*np.dot(np.random.randn(n1, 2), sqrtm(vcov1))
X2 = m2 + 0.3*np.dot(np.random.randn(n2, 2), sqrtm(vcov1))
X3 = m3 + 0.5*np.dot(np.random.randn(n3, 2), sqrtm(vcov2))

X = np.r_[X1, X2, X3]
plt.scatter(X[:, 0], X[:, 1])

# Se agregan puntos outliers
Xout = 2.5 * np.random.randn(10, 2)
X = np.r_[X, Xout]
plt.scatter(X[:, 0], X[:, 1])


# --------------------------------------------------------------------------
# Local Outlier Factor
# Origen: Del curso, class6_book1_outlier_lof.ipynb
# --------------------------------------------------------------------------

# n_neighbors es el k de la fórmula
lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')

# -1 para outliers, 1 para inliers
y_pred = lof.fit_predict(X)

# Puntuaciones de anomalía: cuanto más negativo, más atípico
lofs_scores = lof.negative_outlier_factor_

print("Predicciones (1 = normal, -1 = atípico):", y_pred)
print("Puntuaciones LOF:", lofs_scores)


# Gráfico con círculos proporcionales al score
from matplotlib.legend_handler import HandlerPathCollection

def update_legend_marker_size(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([20])

plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
radius = (lofs_scores.max() - lofs_scores) / (lofs_scores.max() - lofs_scores.min())
scatter = plt.scatter(
    X[:, 0], X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("tight")
plt.xlim((-7.5, 7.5))
plt.ylim((-7.5, 7.5))
plt.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
plt.title("Local Outlier Factor (LOF)")
plt.show()


# --------------------------------------------------------------------------
# Isolation Forest
# Origen: Del curso, class6_book1_outlier_lof.ipynb
# --------------------------------------------------------------------------

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=50, random_state=0).fit(X)

clf_scores = clf.decision_function(X)   # negativo = anómalo
clf_labels = clf.predict(X)             # -1 outlier, 1 inlier

clf.decision_function(X)

plt.scatter(X[:, 0], X[:, 1], c=clf_labels)


# Datos reales: aplicar el modelo a otra base
#!pip install -U ucimlrepo
from ucimlrepo import fetch_ucirepo

lymphography = fetch_ucirepo(id=63)
Xl = lymphography.data.features
yl = lymphography.data.targets

clfl = IsolationForest(n_estimators=50, random_state=0).fit(Xl)
clfl_scores = clf.decision_function(Xl)
clfl_scores
