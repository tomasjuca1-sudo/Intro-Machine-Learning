# ==========================================================================
# TEMA 18: SELECCIÓN DE VARIABLES
# Clase 12, Clase 13
# ==========================================================================


# --------------------------------------------------------------------------
# Método forward con R2 ajustado
# Origen: Del curso, class13_book1_variable_selection.ipynb
# --------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

XTotal, yTotal = datasets.load_diabetes(return_X_y=True)
XTrain, XTest, yTrain, yTest = train_test_split(XTotal, yTotal,
                                                test_size=0.33, random_state=18)

p = len(XTrain[1, ])
datosTrain = len(yTrain)

r2adj = []

for a in range(1, p+1):
    if(a < (p)):
        sfs = SequentialFeatureSelector(LinearRegression(),
                                        n_features_to_select=a,
                                        direction='forward')
        sfs.fit(XTrain, yTrain)
        XTrainSeleccionado = sfs.fit_transform(XTrain, yTrain)
        regAuxiliar = LinearRegression()
        regAuxiliar.fit(XTrainSeleccionado, yTrain)
        r2Modelo = regAuxiliar.score(XTrainSeleccionado, yTrain)
        r2adjModelo = 1-(1-r2Modelo)*(datosTrain-1)/(datosTrain-a-1)
        r2adj.append(r2adjModelo)
    else:
        regAuxiliar = LinearRegression()
        regAuxiliar.fit(XTrain, yTrain)
        r2Modelo = regAuxiliar.score(XTrain, yTrain)
        r2adjModelo = 1-(1-r2Modelo)*(datosTrain-1)/(datosTrain-a-1)
        r2adj.append(r2adjModelo)

plt.plot(range(1, p+1), r2adj)
plt.xlabel('Número de variables')
plt.title('R^2 ajustado')
plt.show()


# --------------------------------------------------------------------------
# Elegir el mejor modelo forward y calcular el MSE
# Origen: Del curso, class13_book1_variable_selection.ipynb
# --------------------------------------------------------------------------

nVariablesSeleccionadas = np.argmax(r2adj)+1   # el +1 corrige el índice base 0

if(nVariablesSeleccionadas < p):
    sfsElegido = SequentialFeatureSelector(LinearRegression(),
                                           n_features_to_select=nVariablesSeleccionadas)
    sfsElegido.fit(XTrain, yTrain)
    XTrainSeleccionado = sfsElegido.fit_transform(XTrain, yTrain)
    modeloFwd = LinearRegression()
    modeloFwd.fit(XTrainSeleccionado, yTrain)
    variablesSeleccionadas = np.array(range(1, p+1))[sfsElegido.get_support()]
else:
    modeloFwd = LinearRegression()
    modeloFwd.fit(XTrain, yTrain)
    variablesSeleccionadas = np.array(range(1, p+1))

mensaje = ['El número de variables seleccionadas es', str(nVariablesSeleccionadas),
           'y corresponden a las columnas:', str(variablesSeleccionadas)]
print(" ".join(mensaje))

# Predicción y MSE. Ojo con el -1 por el cambio de base del índice
prediccionFwd = modeloFwd.predict(XTest[:, variablesSeleccionadas-1])
MSEFwd = np.average(np.square(prediccionFwd-yTest))
print('MSE forward:', MSEFwd)


# --------------------------------------------------------------------------
# Método exhaustivo con mlxtend
# Origen: Del curso, class13_book1_variable_selection.ipynb
# --------------------------------------------------------------------------

from mlxtend.feature_selection import ExhaustiveFeatureSelector

sfse = ExhaustiveFeatureSelector(LinearRegression(),
                                 min_features=1, max_features=p,
                                 scoring="neg_mean_squared_error")
sfse.fit(XTrain, yTrain)

mensaje = ['El número de variables seleccionadas es',
           str(len(sfse.best_feature_names_)),
           'y corresponden a las columnas:',
           str(np.vectorize(int)(np.array(sfse.best_feature_names_))
               + [1]*len(sfse.best_feature_names_))]
print(" ".join(mensaje))

XTrainSeleccionado = sfse.transform(XTrain)
modeloExh = LinearRegression()
modeloExh.fit(XTrainSeleccionado, yTrain)
XTestSeleccionado = sfse.transform(XTest)
prediccionExh = modeloExh.predict(XTestSeleccionado)
MSEExh = np.average(np.square(prediccionExh-yTest))
print('MSE exhaustivo:', MSEExh)


# Comparación de modelos
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_title("Comparación de modelos")
ejeX = ['Método forward', 'Método exhaustivo']
ejeY = [MSEFwd, MSEExh]
ax.bar(ejeX, ejeY)

def addlabels(x, y, plotP):
    for i in range(len(x)):
        plotP.text(i, y[i], y[i])

addlabels(ejeX, ejeY, plt)
plt.show()


# --------------------------------------------------------------------------
# AIC y BIC en Python con statsmodels
# Origen: Equivalente agregado, no viene del material del curso
# --------------------------------------------------------------------------

import statsmodels.api as sm
import numpy as np

# statsmodels requiere agregar la constante a mano
X_const = sm.add_constant(XTrain)
modelo = sm.OLS(yTrain, X_const).fit()

print(modelo.summary())
print("AIC:", modelo.aic)
print("BIC:", modelo.bic)
print("R2 ajustado:", modelo.rsquared_adj)

# Cp de Mallows calculado a mano
modelo_completo = sm.OLS(yTrain, sm.add_constant(XTrain)).fit()
sigma2 = modelo_completo.mse_resid

def cp_mallows(modelo, sigma2, n):
    rss = np.sum(modelo.resid**2)
    d = modelo.df_model + 1
    return (rss + 2*d*sigma2)/n

print("Cp:", cp_mallows(modelo, sigma2, len(yTrain)))
