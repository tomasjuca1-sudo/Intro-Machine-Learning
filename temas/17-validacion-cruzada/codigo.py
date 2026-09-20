# ==========================================================================
# TEMA 17: VALIDACIÓN CRUZADA
# Clase 11, Clase 12
# ==========================================================================


# --------------------------------------------------------------------------
# Las tres formas de particionar
# Origen: Del curso, class11_book2_CV.ipynb
# --------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

XTotal, yTotal = datasets.load_diabetes(return_X_y=True)
datosTotal = len(XTotal)

# 1. Una sola partición
XTrain, XTest, yTrain, yTest = train_test_split(XTotal, yTotal,
                                                test_size=0.33, random_state=0)

# 2. Folds a mano
folds = 10
kf = KFold(n_splits=folds)
kf.get_n_splits(XTotal)

for train_index, test_index in kf.split(XTotal):
    X_trainCV, X_testCV = XTotal[train_index], XTotal[test_index]
    y_trainCV, y_testCV = yTotal[train_index], yTotal[test_index]
    # aquí va el ajuste del modelo y el cálculo del error de este fold

# 3. Métrica automática sobre los folds
from sklearn.model_selection import cross_val_score

neigh = KNeighborsRegressor(n_neighbors=3)

X = XTotal[:150]
y = yTotal[:150]

print(cross_val_score(neigh, X, y, cv=10, scoring="neg_mean_squared_error"))


# --------------------------------------------------------------------------
# Curva de MSE por validación cruzada (ejercicio del notebook resuelto)
# Origen: Solución agregada al ejercicio propuesto en class11_book2_CV.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def curva_mse_cv(XTotal, yTotal, folds):
    n_total = len(XTotal)
    max_vecinos = n_total - int(n_total/folds)   # límite por el tamaño de cada fold

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    lista_k = range(1, max_vecinos)
    mse_por_k = []

    for k in lista_k:
        errores_fold = []
        for train_index, test_index in kf.split(XTotal):
            X_trainCV, X_testCV = XTotal[train_index], XTotal[test_index]
            y_trainCV, y_testCV = yTotal[train_index], yTotal[test_index]

            modelo = KNeighborsRegressor(n_neighbors=k)
            modelo.fit(X_trainCV, y_trainCV)
            prediccion = modelo.predict(X_testCV)

            errores_fold.append(np.average(np.square(prediccion - y_testCV)))

        mse_por_k.append(np.mean(errores_fold))

    plt.plot(list(lista_k), mse_por_k)
    plt.xlabel('Número de vecinos')
    plt.ylabel('MSE por validación cruzada')
    plt.title('Curva de calibración')
    plt.show()

    k_optimo = list(lista_k)[int(np.argmin(mse_por_k))]
    print("k óptimo:", k_optimo, "con MSE:", round(min(mse_por_k), 3))
    return k_optimo, mse_por_k

k_opt, curva = curva_mse_cv(XTotal, yTotal, 10)


# --------------------------------------------------------------------------
# Versión compacta con cross_val_score y LOOCV
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

# Curva de calibración en pocas líneas
lista_k = range(1, 50)
mse_cv = []

for k in lista_k:
    puntajes = cross_val_score(KNeighborsRegressor(n_neighbors=k),
                               XTotal, yTotal, cv=10,
                               scoring="neg_mean_squared_error")
    mse_cv.append(-puntajes.mean())   # el signo negativo se revierte

plt.plot(list(lista_k), mse_cv)
plt.xlabel("k")
plt.ylabel("MSE por CV")
plt.show()

print("k óptimo:", list(lista_k)[int(np.argmin(mse_cv))])


# Leave One Out
loo = LeaveOneOut()
puntajes_loo = cross_val_score(KNeighborsRegressor(n_neighbors=5),
                               XTotal, yTotal, cv=loo,
                               scoring="neg_mean_squared_error")
print("LOOCV MSE:", -puntajes_loo.mean())


# Búsqueda automática del mejor parámetro
rejilla = {'n_neighbors': list(range(1, 50))}
busqueda = GridSearchCV(KNeighborsRegressor(), rejilla, cv=10,
                        scoring="neg_mean_squared_error")
busqueda.fit(XTotal, yTotal)

print(busqueda.best_params_)
print(-busqueda.best_score_)
