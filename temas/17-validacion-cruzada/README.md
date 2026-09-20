# 17. Validación cruzada

**Parte:** Supervisado  
**Clases:** Clase 11, Clase 12  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### El problema con una sola partición

Partir la muestra una sola vez en train y test tiene dos defectos:

1. El MSE de test depende de **qué partición salió**. Con otra semilla da otro número.
2. Se desperdicia información: el modelo final solo se entrena con una fracción de los datos.

### Validación cruzada en K folds

Se parte la muestra en K bloques (folds) del mismo tamaño. Luego, K veces:

1. Se entrena con K-1 folds.
2. Se evalúa en el fold restante.

El estimador final es el promedio de los K errores:

```
              1
CV(K) =  --------- * sum_{j=1}^{K}  MSE_j
              K
```

Cada observación se usa exactamente una vez para evaluar y K-1 veces para entrenar.

### Elección de K

| K | Sesgo del estimador | Varianza | Costo |
|---|---|---|---|
| K pequeño (5) | más sesgo (entrena con menos datos) | menos varianza | barato |
| K grande (10) | menos sesgo | más varianza | medio |
| K = n (LOOCV) | sesgo mínimo | varianza alta | caro |

El estándar de la práctica es **K = 5 o K = 10**. El código del curso usa 10.

### Para qué se usa

Principalmente para **calibrar el parámetro de flexibilidad**: se recorre una rejilla de valores del
parámetro, se calcula el CV para cada uno, y se escoge el que minimiza el error. Es lo que hacen por
dentro `RidgeCV`, `LassoCV`, `ElasticNetCV` y `cv.glmnet`.

### Advertencia

Toda transformación que use información de la respuesta o de toda la muestra (escalamiento,
selección de variables) debe hacerse **dentro** de cada fold, no antes. Si se hace antes, la
información del fold de evaluación se filtra al entrenamiento y el error estimado sale
optimistamente bajo.

---

## Cómo se relaciona el código con el tema

El notebook muestra **tres niveles** de la misma idea, de más manual a más automático.

**Nivel 1, una sola partición.**

```
XTrain, XTest, yTrain, yTest = train_test_split(XTotal, yTotal, test_size=0.33, random_state=0)
```

`test_size` es la proporción que va a evaluación. `random_state` fija la semilla para que la
partición sea reproducible.

**Nivel 2, los folds a mano.**

```
kf = KFold(n_splits=folds)
kf.get_n_splits(XTotal)

for train_index, test_index in kf.split(XTotal):
    X_trainCV, X_testCV = XTotal[train_index], XTotal[test_index]
    y_trainCV, y_testCV = yTotal[train_index], yTotal[test_index]
```

Importante: `kf.split()` devuelve **índices**, no datos. El cuerpo del loop es donde iría el ajuste
del modelo y el cálculo del error de ese fold. En el notebook el loop está vacío a propósito, como
plantilla.

**Nivel 3, automático.**

```
print(cross_val_score(neigh, X, y, cv=10, scoring="neg_mean_squared_error"))
```

Devuelve un vector con el puntaje de cada fold. Ojo con el prefijo **neg_**: sklearn siempre
maximiza, así que reporta el MSE con signo negativo. Para obtener el MSE hay que cambiarle el signo:
`-cross_val_score(...).mean()`.

El ejercicio final del notebook pide escribir una función que grafique la curva de MSE por validación
cruzada contra el número de vecinos. Abajo está resuelto.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Las tres formas de particionar | Del curso, class11_book2_CV.ipynb |
| Python | Curva de MSE por validación cruzada (ejercicio del notebook resuelto) | Solución agregada al ejercicio propuesto en class11_book2_CV.ipynb |
| Python | Versión compacta con cross_val_score y LOOCV | Equivalente agregado, no viene del material del curso |
| R | Validación cruzada en R | Equivalente agregado, no viene del material del curso |
