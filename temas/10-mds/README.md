# 10. Multidimensional Scaling (MDS)

**Parte:** Reducción de dimensiones  
**Clases:** Clase 7, Clase 8  
**Archivos:** [`codigo.py`](codigo.py) | [`codigo.R`](codigo.R)

---

## Explicación del tema

### La intuición fundamental

Suponga que se tienen las distancias entre un grupo de observaciones, contenidas en una matriz:

```
D = [ d_ij ]  en R^{n x n}
```

pero **no se tienen las coordenadas originales**, es decir, se desconoce la matriz X.
¿Cómo se pueden reconstruir las observaciones? En particular, ¿cómo hacer un mapa en 2D?

La clave está en la relación entre las distancias y los **productos punto**:

```
D = [ d_ij ]  en R^{n x n}
B = [ <x_i, x_j> ] = X X^T  en R^{n x n}
```

Conociendo D se puede recuperar B mediante doble centrado, y de B se recuperan coordenadas
haciendo descomposición espectral. Ese es el MDS clásico.

### Los tres sabores

**1. MDS clásico (cmdscale).** Usa distancias euclidianas y doble centrado. Es **equivalente a PCA**
cuando la matriz de distancias es euclidiana: da las mismas coordenadas salvo rotación y reflexión.
La diferencia es la entrada: PCA recibe X, MDS recibe D.

**2. MDS métrico.** Se usa cuando las distancias no son euclidianas pero sí son cuantitativas, por
ejemplo la distancia de Gower para datos mixtos. Busca coordenadas que aproximen los valores de las
distancias minimizando una función de estrés.

**3. MDS no métrico (isoMDS, Kruskal).** Se usa cuando solo importa el **orden** de las distancias,
no sus valores. Típico con datos de percepción, similitud o votaciones. Minimiza el estrés sobre una
transformación monótona de las distancias.

### Salidas y diagnósticos

- `mds$points`: las coordenadas nuevas (la matriz Z).
- `mds$eig`: los eigenvalores. Eigenvalores negativos grandes indican que la matriz de distancias
  no es euclidiana. La proporción de los primeros d eigenvalores indica la calidad del mapa.
- En el no métrico, el **estrés** mide el ajuste. Menos de 5 por ciento es excelente, más de 20 por
  ciento es malo.

### Ejemplo canónico

`eurodist` contiene las distancias por carretera entre ciudades europeas. MDS reconstruye el mapa de
Europa a partir de esa tabla, sin conocer una sola coordenada geográfica. El mapa sale rotado o
reflejado (por eso el código escribe `y <- -mds$points[,2]`, para poner el norte arriba). Esa
ambigüedad de rotación y reflexión es inherente al método.

---

## Cómo se relaciona el código con el tema

El código del curso muestra los tres casos, uno tras otro.

**Caso 1, clásico:** `mds = cmdscale(eurodist, k = 2, eig = TRUE)`.
El argumento `k` es la dimensión destino y `eig = TRUE` pide los eigenvalores para diagnosticar.
La línea `y <- -mds$points[, 2]` invierte el segundo eje **solo por estética**, para que el norte
quede arriba. El gráfico se arma con `type = "n"` (no dibujar puntos) y luego `text()` para poner
los nombres de las ciudades.

**Caso 2, métrico con datos mixtos:** aquí está el paso clave del tema.

```
dis = daisy(xx, metric = "gower")   # matriz de distancias para datos mixtos
mds = cmdscale(dis, k = 2, eig = TRUE)
```

La función `daisy` del paquete `cluster` convierte una base con variables categóricas y continuas en
una matriz de distancias. Es exactamente la misma idea del tema 04 (K-Medoides con Gower): si el
método solo necesita distancias, se puede aplicar a cualquier tipo de dato.

**Caso 3, no métrico:** `mds = isoMDS(voting)` del paquete MASS, sobre datos de votaciones donde
solo el orden de similitud tiene sentido.

---

## Índice de bloques de código

| Lenguaje | Bloque | Origen |
|---|---|---|
| Python | MDS clásico, métrico y no métrico con sklearn | Equivalente agregado. El código de MDS del curso está en R |
| Python | MDS clásico a mano (doble centrado) | Equivalente agregado, no viene del material del curso |
| Python | Distancia de Gower en Python para MDS con datos mixtos | Equivalente agregado, no viene del material del curso |
| R | MDS clásico, métrico y no métrico (código del curso) | Del curso, code3_class7_MDS.txt |
| R | Diagnóstico de calidad del MDS | Equivalente agregado, no viene del material del curso |
