# 02. K-Means

**Parte:** Clustering  
**Clases:** Clase 2, Clase 3  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Qué resuelve

Se fija un número de clusters K y se busca asignar cada punto a un grupo de forma que se minimice
la suma de distancias de los puntos al centro de su grupo (máxima coherencia interna):

```
min  sum_{k=1}^{K}  sum_{x_i en cluster k}  || x_i - c_k ||^2
```

El centroide es el promedio muestral del grupo:

```
c_k = (1 / n_k) * sum_{x_i en cluster k} x_i = xbar_k
```

### Dos equivalencias que suele preguntar

**1. Distancias al centro igual a distancias por pares.** Al usar el promedio como centroide:

```
sum_{x en cluster k} ||x - xbar_k||^2  =  (1 / 2 n_k) * sum_{x,y en cluster k} ||x - y||^2
```

**2. Descomposición ANOVA.** La suma total de cuadrados se parte en dos:

```
SST = BSS + WSS
sum_i ||x_i - Xbar||^2 = sum_k n_k ||xbar_k - Xbar||^2 + sum_k sum_{x en k} ||x - xbar_k||^2
```

SST no depende de la partición. Entonces **minimizar WSS es equivalente a maximizar BSS**.
Eso conecta coherencia interna con separación entre grupos en un solo criterio.

### Complejidad

El problema exacto es **NP-Hard** cuando p > 2 y no escala en n. Por eso se usa una heurística
**greedy**: se alternan dos pasos hasta que el SSE deje de cambiar.

1. Dado el conjunto de centros, asignar cada punto al centro más cercano.
2. Dada la asignación, recalcular los centros como promedio.

Como puede caer en mínimos locales, se repite con varias inicializaciones aleatorias y se escoge
la mejor. En sklearn eso es `n_init`.

### Problemas del método

1. La separación se mide por la distancia entre centros, pero hay riesgo de traslape en las
   fronteras internas.
2. Los clusters tienden a tener forma regular (bolas en p dimensiones). Con formas irregulares
   falla.
3. **No es robusto** a datos atípicos, porque el promedio se corre con un solo outlier.
4. **Depende de la escala**. Si las variables están en unidades distintas, la variable con mayor
   rango domina la distancia. Por eso se estandariza antes de correr el algoritmo.

### Selección de K

1. **Método del codo**: se gráfica el SSE (o la inercia) contra K y se busca el punto donde la
   ganancia marginal se aplana.
2. **Métricas externas al algoritmo**: Silhouette, Calinski-Harabasz, Davies-Bouldin. Ver tema 03.

---

## Cómo se relaciona el código con el tema

El notebook implementa el algoritmo **dos veces**: a mano y con sklearn. Vale la pena tener las dos,
porque un parcial puede pedir el paso intermedio.

Mapa entre teoría y código de la versión manual:

| Concepto | Objeto en el código |
|---|---|
| Asignación z_ik | `assign_matrix`, matriz n por K de ceros y unos |
| n_k | `n_group = sum(assign_matrix)` |
| Centroides c_k | `means = np.matmul(X.T, assign_matrix) / n_group`, matriz p por K |
| Distancias al centro | `dista`, matriz n por K |
| WSS o SSE | `np.sum(np.diag(np.matmul(dista.T, assign_matrix)))` |
| Paso de asignación | `assign_matrix[i, np.argmin(dista[i,:])] = 1` |
| Criterio de parada | `change = sse_hist[-2] - sse` comparado contra `tol` |

Detalle útil: `np.diag(dista.T @ assign_matrix)` toma, para cada punto, solo la distancia al grupo
al que quedó asignado. Sumar esa diagonal es exactamente el WSS.

En la versión con sklearn, `kmeans.cluster_centers_` son los centroides (una fila por grupo, o sea
la transpuesta de `means`) y `kmeans.labels_` son las etiquetas. El escalamiento se hace con
`scale()` de `sklearn.preprocessing` antes de ajustar.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | K-Means paso a paso (implementación manual) | Del curso, class2_book1_kmeans.ipynb |
| Python | K-Means con scikit-learn y escalamiento | Del curso, class2_book1_kmeans.ipynb |
| Python | Método del codo para escoger K | Equivalente agregado, no viene del material del curso |
| R | K-Means en R con escalamiento y codo | Equivalente agregado, no viene del material del curso |
