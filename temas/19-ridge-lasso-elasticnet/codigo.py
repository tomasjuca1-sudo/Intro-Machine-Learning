# ==========================================================================
# TEMA 19: RIDGE, LASSO Y ELASTIC NET
# Clase 14
# ==========================================================================


# --------------------------------------------------------------------------
# Ridge con calibración por CV
# Origen: Del curso, class14_book1_penalty.ipynb
# --------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import ElasticNetCV, ElasticNet
import matplotlib.pyplot as plt

XTotal, yTotal = datasets.load_diabetes(return_X_y=True)
XTrain, XTest, yTrain, yTest = train_test_split(XTotal, yTotal,
                                                test_size=0.33, random_state=0)

# Rejilla logarítmica de valores de lambda
n_alphas = 200
alphasCalibrar = np.logspace(-10, 2, n_alphas)

modeloRidge = RidgeCV(alphas=alphasCalibrar, store_cv_values=True).fit(XTrain, yTrain)

print('El parámetro de penalización óptimo es:')
print(modeloRidge.alpha_)
print('El valor del R^2 es:')
print(modeloRidge.score(XTrain, yTrain))
print('Los coeficientes estimados son:')
print(modeloRidge.coef_)

# Trayectoria de los coeficientes al variar lambda
coefs = []
for a in alphasCalibrar:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(XTrain, yTrain)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphasCalibrar, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Parámetro de penalización')
plt.ylabel('Coeficientes')
plt.title('Penalización Ridge')
plt.axis('tight')
plt.show()

prediccionRidge = modeloRidge.predict(XTest)
MSERidge = np.average(np.square(prediccionRidge-yTest))
print('MSE Ridge:', MSERidge)


# --------------------------------------------------------------------------
# LASSO
# Origen: Del curso, class14_book1_penalty.ipynb
# --------------------------------------------------------------------------

n_alphas = 200
alphasCalibrar = np.logspace(-10, 2, n_alphas)

modeloLASSO = LassoCV(alphas=alphasCalibrar).fit(XTrain, yTrain)

print('El parámetro de penalización es:')
print(modeloLASSO.alpha_)
print('El valor del R^2 es:')
print(modeloLASSO.score(XTrain, yTrain))
print('Los coeficientes estimados son:')
print(modeloLASSO.coef_)   # algunos seran exactamente cero

coefs = []
for a in alphasCalibrar:
    lasso = Lasso(alpha=a, fit_intercept=False, max_iter=10000)
    lasso.fit(XTrain, yTrain)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphasCalibrar, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Parámetro de penalización')
plt.ylabel('Coeficientes')
plt.title('Penalización LASSO')
plt.axis('tight')
plt.show()

prediccionLASSO = modeloLASSO.predict(XTest)
MSELASSO = np.average(np.square(prediccionLASSO-yTest))
print('MSE LASSO:', MSELASSO)

# Variables que sobrevivieron
print("Variables seleccionadas:", np.where(modeloLASSO.coef_ != 0)[0])


# --------------------------------------------------------------------------
# Elastic Net y comparación final
# Origen: Del curso, class14_book1_penalty.ipynb
# --------------------------------------------------------------------------

n_alphas = 200
alphasCalibrar = np.logspace(-10, 2, n_alphas)

modeloElastic = ElasticNetCV(alphas=alphasCalibrar, cv=5,
                             random_state=0).fit(XTrain, yTrain)

print('El parámetro de penalización es:')
print(modeloElastic.alpha_)
print('El valor del R^2 es:')
print(modeloElastic.score(XTrain, yTrain))
print('Los coeficientes estimados son:')
print(modeloElastic.coef_)

coefs = []
for a in alphasCalibrar:
    elasticNet = ElasticNet(alpha=a, fit_intercept=False, max_iter=10000)
    elasticNet.fit(XTrain, yTrain)
    coefs.append(elasticNet.coef_)

ax = plt.gca()
ax.plot(alphasCalibrar, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Parámetro de penalización')
plt.ylabel('Coeficientes')
plt.title('Penalización Elastic Net')
plt.axis('tight')
plt.show()

prediccionElastic = modeloElastic.predict(XTest)
MSEElastic = np.average(np.square(prediccionElastic-yTest))
print('MSE Elastic Net:', MSEElastic)


# Comparación de los tres modelos
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_title("Comparación de modelos")
ejeX = ['LASSO', 'Ridge', 'Elastic Net']
ejeY = [MSELASSO, MSERidge, MSEElastic]
ax.bar(ejeX, ejeY)

def addlabels(x, y, plotP):
    for i in range(len(x)):
        plotP.text(i, y[i], y[i])

addlabels(ejeX, ejeY, plt)
plt.show()


# --------------------------------------------------------------------------
# Calibrar lambda y l1_ratio al tiempo
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# ElasticNetCV puede calibrar los dos parámetros a la vez
modelo = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                      alphas=np.logspace(-4, 2, 100),
                      cv=10, random_state=0, max_iter=10000)
modelo.fit(XTrain, yTrain)

print("lambda escogido:", modelo.alpha_)
print("l1_ratio escogido:", modelo.l1_ratio_)
print("Coeficientes en cero:", np.sum(modelo.coef_ == 0))

# Con escalamiento dentro del pipeline (evita filtrar información entre folds)
tuberia = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                 alphas=np.logspace(-4, 2, 100),
                 cv=10, random_state=0, max_iter=10000)
)
tuberia.fit(XTrain, yTrain)

prediccion = tuberia.predict(XTest)
print("MSE:", np.average(np.square(prediccion - yTest)))
