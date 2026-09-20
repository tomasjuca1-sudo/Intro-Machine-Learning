# 18. Selección de variables

**Parte:** Supervisado  
**Clases:** Clase 12, Clase 13  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Por qué se hace

En un modelo de regresión lineal, la **cantidad de variables predictoras es el parámetro de
flexibilidad**:

- Más variables: más flexibilidad, más varianza, menos sesgo.
- Menos variables: menos flexibilidad, menos varianza, más sesgo.

Por eso hay que encontrar una forma sistemática de escoger el conjunto de variables que mejor
predice la respuesta.

### Método exhaustivo

Compara **todos** los modelos posibles y elige el mejor bajo cierta métrica. Garantiza encontrar el
óptimo. El número de modelos crece de forma **exponencial**:

```
2^p - 1  modelos
```

Con p = 20 son más de un millón. Se vuelve impracticable rápido.

### Método forward

Heurística. Empieza comparando todos los modelos con una sola variable. Luego agrega variables una a
una, **dejando fijas** las que ya entraron en iteraciones anteriores. No garantiza el óptimo, pero
es mucho más eficiente. El número de modelos crece de forma **cuadrática**:

```
p(p+1) / 2  modelos
```

Con p = 20 son 210 modelos. También existe el método **backward**, que empieza con todas y va
quitando.

### Métricas para comparar modelos con distinto número de variables

No se puede usar el R^2 crudo ni el RSS, porque siempre mejoran al agregar variables. Hay que
penalizar el tamaño del modelo:

**R^2 ajustado.** Se busca el **máximo**.

```
                  (1 - R^2)(n - 1)
R2_adj = 1 -  ------------------------
                   n - p - 1
```

**Cp de Mallows.** Se busca el **mínimo**. Estima el error de predicción.

```
Cp = (1/n) * ( RSS + 2 * d * sigma_hat^2 )
```

**AIC.** Se busca el **mínimo**. Penaliza con 2 por parámetro.

**BIC.** Se busca el **mínimo**. Penaliza con log(n) por parámetro, así que castiga más los modelos
grandes cuando n es grande. **BIC tiende a escoger modelos más pequeños que AIC.**

### Regla práctica para el parcial

| Métrica | Dirección | Penalización |
|---|---|---|
| R^2 ajustado | máximo | moderada |
| Cp de Mallows | mínimo | 2 por parámetro |
| AIC | mínimo | 2 por parámetro |
| BIC | mínimo | log(n) por parámetro, la más fuerte |
| MSE en test o CV | mínimo | ninguna, se mide directo |

---

## Cómo se relaciona el código con el tema

### En Python

Dos funciones distintas, una por método:

- **Forward**: `SequentialFeatureSelector(LinearRegression(), n_features_to_select=a,
  direction='forward')` de `sklearn.feature_selection`.
- **Exhaustivo**: `ExhaustiveFeatureSelector(LinearRegression(), min_features=1, max_features=p,
  scoring="neg_mean_squared_error")` del paquete **mlxtend**, que no es sklearn.

El notebook implementa el forward con un loop explícito sobre el número de variables `a`, calculando
el R2 ajustado en cada iteración:

```python
r2Modelo = regAuxiliar.score(XTrainSeleccionado, yTrain)
r2adjModelo = 1-(1-r2Modelo)*(datosTrain-1)/(datosTrain-a-1)
```

Ese es exactamente el R2 ajustado de la fórmula. El `if(a < p)` existe porque cuando se piden todas
las variables no hay nada que seleccionar, se ajusta el modelo completo directo.

Después `nVariablesSeleccionadas = np.argmax(r2adj)+1` toma el máximo. El **+1** es porque el índice
0 corresponde al modelo con 1 variable.

Cuidado con el indexado al predecir: `XTest[:, variablesSeleccionadas-1]`. El vector
`variablesSeleccionadas` está en base 1 (se construyó con `range(1, p+1)`), y numpy indexa en base 0.

### En R

Una sola función hace todo: **`regsubsets`** del paquete `leaps`.

```
reg_subset = regsubsets(Salary~., Hitters, nvmax=12, method="exhaustive")
```

- La fórmula `Y~.` significa respuesta contra todas las demás variables.
- `nvmax` es el máximo número de variables (el L del comentario del profesor).
- `method` cambia entre `"exhaustive"`, `"forward"` y `"backward"`. **Ese es el único cambio entre
  los dos métodos.**

El resultado se lee con `summary()`, y de ahí salen las métricas:

```
reg_sub_summary$cp      # Cp de Mallows de cada mejor modelo por tamaño
reg_sub_summary$bic     # BIC
reg_sub_summary$adjr2   # R2 ajustado
```

Se grafican con `plot(..., type="b")` y se busca el mínimo (Cp, BIC) o el máximo (adjr2). La función
`which.min()` da la posición.

Detalle: `Hitters = na.omit(Hitters)` es obligatorio, porque `regsubsets` no maneja datos faltantes.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Método forward con R2 ajustado | Del curso, class13_book1_variable_selection.ipynb |
| Python | Elegir el mejor modelo forward y calcular el MSE | Del curso, class13_book1_variable_selection.ipynb |
| Python | Método exhaustivo con mlxtend | Del curso, class13_book1_variable_selection.ipynb |
| Python | AIC y BIC en Python con statsmodels | Equivalente agregado, no viene del material del curso |
| R | regsubsets: exhaustivo y forward (código del curso) | Del curso, code1_class13.txt |
| R | Extraer el mejor modelo y sus coeficientes | Equivalente agregado, no viene del material del curso |
| R | Selección con MSE en test | Del curso, code2_class14.txt (primera parte) |
