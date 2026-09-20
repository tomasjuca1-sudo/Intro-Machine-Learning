# 06. DBSCAN

**Parte:** Clustering  
**Clases:** Clase 4  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### Idea central

Los métodos basados en densidades usan como criterio la **densidad local de puntos por volumen**.
Un cluster es una aglomeración de puntos densamente conectados, **sin necesidad de que todos los
puntos del grupo estén cercanos entre si**. Por eso detecta formas irregulares (círculos
concentricos, líneas, espirales) que K-Means no puede.

DBSCAN significa Density-Based Spatial Clustering of Applications with Noise. Además de encontrar
los clusters, **reconoce los puntos atípicos** que no pertenecen a ningún grupo.

### Los dos parámetros

Se construye una bola (vecindad) alrededor de cada punto:

1. **eps (epsilon)**: el radio de la bola donde se buscan vecinos.
2. **MinPts (`min_samples`)**: el mínimo número de puntos dentro de la vecindad para considerar
   que el punto está dentro de un grupo.

### Clasificación de los puntos

| Tipo | Condición |
|---|---|
| **Core** (interno) | su vecindad de radio eps tiene al menos MinPts vecinos |
| **Border** (borde) | tiene vecinos que son core, pero su propio número de vecinos es menor a MinPts |
| **Noise** (ruido, outlier) | no es core ni border |

En sklearn, los puntos de ruido reciben la etiqueta **-1**.

### Ventajas y desventajas

**Ventajas.** Detecta formas irregulares. No hay que fijar K de antemano. Identifica outliers de
forma explícita. Es robusto al ruido.

**Desventajas.** Los parámetros son difíciles de calibrar. Falla cuando los clusters tienen
densidades muy distintas, porque un solo par (eps, MinPts) no sirve para todos. Se degrada en
dimensiones altas porque las distancias se vuelven parecidas entre si.

### Cómo escoger eps

Procedimiento del curso: para un k dado, se calcula la distancia de cada punto a su k-ésimo vecino
más cercano, se ordenan esas distancias de menor a mayor y se grafican. El codo de esa curva sugiere
un eps razonable. La respuesta depende de **qué proporción de puntos outliers se permite**: si se
corta bajo, más puntos quedan como ruido.

---

## Cómo se relaciona el código con el tema

La función `distancia_kveci(X, k)` del notebook es la herramienta de calibración, no parte del
algoritmo. Lo que hace:

- `NearestNeighbors(n_neighbors=k).fit(X)` y luego `kneighbors(X)` devuelve, para cada punto, las
  distancias a sus k vecinos más cercanos.
- `distances[:, 1:]` descarta la primera columna porque **el vecino más cercano de un punto es el
  punto mismo, a distancia cero**. Ese detalle es fácil de olvidar.
- `.reshape(-1)` aplana todo en un solo vector.

Luego `plt.plot(sorted(distancia_kveci(X, 5)))` gráfica las distancias ordenadas. El codo de esa
curva es el candidato a eps.

Después se ajusta el modelo: `DBSCAN(eps=0.6, min_samples=6).fit(X)` y `dbsca.labels_` da las
etiquetas, donde -1 marca ruido.

El segundo bloque del notebook simula dos círculos concentricos más ruido uniforme. Es el ejemplo
canónico de por qué DBSCAN gana: K-Means partiría los círculos por la mitad, DBSCAN los separa
como dos anillos y manda el ruido a la clase -1.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | Calibración de eps y ajuste de DBSCAN | Del curso, class4_book2_DBSCAN.ipynb |
| Python | Datos con formas irregulares (círculos concentricos más ruido) | Del curso, class4_book2_DBSCAN.ipynb |
| Python | Resumen de la solución: número de clusters y de outliers | Equivalente agregado, no viene del material del curso |
| R | DBSCAN en R con dbscan y kNNdistplot | Equivalente agregado, no viene del material del curso |
