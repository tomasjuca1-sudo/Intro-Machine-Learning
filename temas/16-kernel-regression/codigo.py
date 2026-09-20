# ==========================================================================
# TEMA 16: KERNEL REGRESSION Y K VECINOS
# Clase 10, Clase 11
# ==========================================================================


# --------------------------------------------------------------------------
# Implementación de kernel regression
# Origen: Del curso, class11_book1.ipynb
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def kernelreg(xstar, X, Y, h):
    pred = np.repeat(0.0, len(xstar))
    for i in range(len(xstar)):
        dist = list()
        dist = abs(xstar[i]-X)
        zz = np.where(dist <= h, Y, 0)
        pred[i] = sum(zz)/max(sum(np.where(dist <= h, 1, 0)), 0.1)
    return(pred)


xstar = np.arange(1.0, 5.0, .01)

# h grande: mucha suavidad, hay SESGO
yyy = kernelreg(xstar, x_train, y_train, .4)
plt.plot(x_train, y_train, 'o', ms=2, color="k")     # datos en negro
plt.plot(points, f_mean(points), "r", linewidth='3') # función óptima en rojo
plt.plot(xstar, yyy, 'b', color="b", linewidth='2')  # función estimada en azul

# h pequeño: mucha VARIANZA
yyy = kernelreg(xstar, x_train, y_train, 0.03)
plt.plot(x_train, y_train, 'o', ms=2, color="k")
plt.plot(points, f_mean(points), "r", linewidth='3')
plt.plot(xstar, yyy, 'b', color="b", linewidth='2')


# --------------------------------------------------------------------------
# Calibración de h y modelo final
# Origen: Del curso, class11_book1.ipynb
# --------------------------------------------------------------------------

hh = np.arange(0.01, 1, 0.002)

mse_train = np.repeat(0.0, len(hh))
mse_test = np.repeat(0.0, len(hh))

for i in range(len(hh)):
    fhat_train = kernelreg(x_train, x_train, y_train, hh[i])
    fhat_test = kernelreg(x_test, x_train, y_train, hh[i])
    mse_train[i] = (sum((fhat_train-y_train)**2)/n_train)
    mse_test[i] = sum((fhat_test-y_test)**2)/(n-n_train)

plt.plot(hh, mse_train, "r", linewidth='3')
plt.plot(hh, mse_test, "b", linewidth='3')

h_opt = hh[np.argmin(mse_test)]
h_opt

mse_opt = min(mse_test)
mse_opt

# Modelo ya calibrado
yyy = kernelreg(xstar, x_train, y_train, h_opt)
plt.plot(x_train, y_train, 'o', ms=2, color="k")
plt.plot(points, f_mean(points), "r", linewidth='3')
plt.plot(xstar, yyy, 'b', color="b", linewidth='2')


# --------------------------------------------------------------------------
# K vecinos con sklearn y MSE en test
# Origen: Del curso, class11_book2_CV.ipynb
# --------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

XTotal, yTotal = datasets.load_diabetes(return_X_y=True)
datosTotal = len(XTotal)

XTrain, XTest, yTrain, yTest = train_test_split(XTotal, yTotal,
                                                test_size=0.33, random_state=0)

vecinos = 3
neigh = KNeighborsRegressor(n_neighbors=vecinos)
neigh.fit(XTrain, yTrain)

prediccionKVecinos = neigh.predict(XTest)

MSEKvecinos = np.average(np.square(prediccionKVecinos-yTest))
mensaje = ['El MSE en test del modelo de K vecinos con k=', str(vecinos),
           'es igual a', str(MSEKvecinos)]
print(" ".join(mensaje))
