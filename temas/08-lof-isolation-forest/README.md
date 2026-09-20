# 08. LOF e Isolation Forest

**Parte:** Outliers  
**Clases:** Clase 6  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Local Outlier Factor (LOF)

Método basado en **densidades**. La idea fundamental es comparar la densidad local de una
observación con las densidades de sus vecinos. Es local: un punto puede ser normal para un grupo
denso y atípico para uno disperso.

Se define la densidad local alcanzable (local reachability density):

```
                        k
lrd_k(x) = ------------------------------
            sum_{y en N_k(x)}  RD_k(x, y)
```

donde N_k(x) es el conjunto de los k vecinos más cercanos a x, y RD_k es la distancia de
alcanzabilidad (reachability distance), que suaviza distancias muy pequeñas.

El factor LOF es la razón entre la densidad de los vecinos y la densidad propia:

```
                1        sum_{y en N_k(x)}  lrd_k(y)
LOF(x) = ------------- *  ---------------------------
          k * lrd_k(x)                 1
```

Interpretación:

- **LOF cerca de 1**: el punto tiene densidad parecida a la de sus vecinos. Normal.
- **LOF mucho mayor que 1**: la densidad del punto es mucho menor que la de sus vecinos. Outlier.
- **LOF menor que 1**: el punto está en una zona más densa que sus vecinos.

La ventaja frente a Mahalanobis es que **no asume una sola nube**. Detecta outliers locales.

### Isolation Forest (iForest)

Método basado en ML supervisado (árboles). La idea fundamental es hacer particiones del espacio X
**aleatorias y paralelas a los ejes** hasta aislar alguna observación.

Razonamiento central: los puntos anómalos son **más fáciles de separar** del resto de la muestra,
es decir, requieren menos particiones.

```
h(x) = número de particiones necesarias para aislar x
```

Se promedia h(x) sobre muchos árboles y se normaliza en un score:

```
s(x, m) cerca de 1    ->  Outlier
s(x, m) cerca de 0.5  ->  Normal
```

Ventajas: escala muy bien en n y en p, no necesita calcular distancias, es rápido.

### Cuándo usar cuál

| Situación | Método |
|---|---|
| Una sola nube, pocas dimensiones | Mahalanobis |
| Varios grupos con densidades distintas | LOF |
| Muchos datos o muchas dimensiones | Isolation Forest |

---

## Cómo se relaciona el código con el tema

### LOF

```
lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')
y_pred = lof.fit_predict(X)
lofs_scores = lof.negative_outlier_factor_
```

Tres cosas para no confundir en el parcial:

- `y_pred` devuelve **-1 para outliers y 1 para inliers**.
- `negative_outlier_factor_` es el LOF con **signo cambiado**. Cuanto más negativo, más atípico.
  Un punto normal da un valor cercano a -1.
- `n_neighbors` es el k de la fórmula. Es el parámetro que define qué tan local es el análisis.

El gráfico del notebook dibuja círculos cuyo radio es proporcional al score normalizado:
`radius = (max - scores) / (max - min)`. A mayor círculo, más atípico el punto.

### Isolation Forest

```
clf = IsolationForest(n_estimators=50, random_state=0).fit(X)
clf_scores = clf.decision_function(X)
clf_labels = clf.predict(X)
```

- `n_estimators` es el número de árboles del bosque.
- `decision_function(X)` da el score: **valores negativos son anómalos**, positivos normales.
  Ojo, sklearn invierte el signo respecto a la s(x,m) de la teoría.
- `predict(X)` da -1 y 1, igual que LOF.
- A diferencia de LOF, IsolationForest sí se puede ajustar en una muestra y aplicar a otra, porque
  guarda el modelo. Por eso en el notebook se ajusta con `X` y se evalúa sobre `Xl`.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Datos simulados con tres grupos más outliers | Del curso, class6_book1_outlier_lof.ipynb |
| Python | Local Outlier Factor | Del curso, class6_book1_outlier_lof.ipynb |
| Python | Isolation Forest | Del curso, class6_book1_outlier_lof.ipynb |
| R | LOF e Isolation Forest en R | Equivalente agregado, no viene del material del curso |
